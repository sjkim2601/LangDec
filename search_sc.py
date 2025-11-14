from typing import Any, Literal, List

import torch

from interfaces import BaseGenerator, BasePRM

import random
import numpy as np
import os
import re
import logging
import torch.nn.functional as F

from collections import defaultdict

level = logging.INFO
if os.getenv('DEBUG', False):
    level = logging.DEBUG

# 로깅 설정
logging.basicConfig(
    level=level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bootstrap_search.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def aggregate(vals, agg_method):
    if agg_method == "min":
        aggregate_scores, _ = torch.min(vals, dim=-1)
    elif agg_method == "mean":
        aggregate_scores = torch.mean(vals, dim=-1)
    elif agg_method == "sum":
        aggregate_scores = torch.sum(vals, dim=-1)
    elif agg_method == "last":
        aggregate_scores = vals[:, -1]
    elif agg_method == "prod":
        aggregate_scores = torch.cumprod(vals, dim=1)[:, -1]
    else:
        raise NotImplementedError(
            f"{agg_method} aggregation is not implemented."
        )
    return aggregate_scores


class SelfConsistencySearch:
    def __init__(
        self,
        method,
        generator: BaseGenerator,
        prm: BasePRM,

        temp_update_rule=None,
        max_trials: int = None,
        score_aggregation: Literal["min", "mean", "last", 'prod'] = "min",
    ):
        self.method = method
        self.generator = generator
        self.prm = prm

        self.temp_update_rule = temp_update_rule
        self.max_trials = max_trials
        self.trials = 0

        self.score_aggregation = score_aggregation

        self.return_all_steps = True

        self.init_number_of_beams = 1

    def compute_step_scores(self, responses: list, prm_state):
        score_tok = getattr(self.prm, "score_token", None)
        responses = [r.replace("\n\n## Step", f"{score_tok}## Step") for r in responses]
        return self.prm(responses, prm_state, return_all_steps=self.return_all_steps)

    # def _update_temperature(self):
    #     if self.temp_update_rule is None:
    #         return None
    #     else:
    #         # TODO
    #         # self.generator.temperature = ...
    #         raise NotImplementedError()
        

    def _update_temperature(self):
        """Ultra-simple temperature updater for Math-500 (no state, few lines)."""
        import math
        from collections import Counter

        # --- read inputs (fall back safely) ---
        T      = float(getattr(self, "temperature", 0.8))
        t_min  = getattr(self, "t_min", 0.2)
        t_max  = getattr(self, "t_max", 1.8)
        answers = getattr(self, "answers", None) or getattr(self, "sc_answers", None) or []
        scores  = getattr(self, "prm_scores", None) or getattr(self, "scores_prm", None)

        # --- quick stats: entropy frac, unique frac, PRM margin ---
        K = max(1, len(answers))
        cnt = Counter(answers) if answers else Counter({"_": K})
        def Hfrac(v):
            tot = sum(v); 
            if K <= 1 or tot == 0: return 0.0
            h = 0.0
            for c in v:
                if c>0:
                    p = c/tot
                    h -= p*math.log(max(p,1e-12))
            return h / math.log(K)
        Hf = Hfrac(list(cnt.values()))
        Uf = len(cnt)/K
        if scores and len(scores) >= 2:
            s = sorted(scores, reverse=True)
            M = max(0.0, s[0]-s[1])
        else:
            M = 0.0

        # --- targets (tune if needed) ---
        tH, tU, tM = 0.55, 0.55, 0.10

        # --- single-step proportional control on log(T) ---
        dlogT = 1.0*(tH - Hf) + 0.8*(tU - Uf) + 0.6*(tM - M)
        dlogT = max(-0.20, min(0.20, dlogT))   # clamp for stability

        # --- apply & clamp ---
        T = T * math.exp(dlogT)
        T = max(t_min, min(t_max, T))
        self.temperature = float(T)
        return self.temperature

    def __call__(self, question: str):
        input_ids_question = self.generator.encode(question)
        gen_state_question = self.generator.init_state(input_ids_question)
        prm_state_question = self.prm.init_state(question)

        input_ids_question = input_ids_question.repeat(self.init_number_of_beams, 1)
        gen_state_question = self.generator.inflate_state(gen_state_question, self.init_number_of_beams)
        prm_state_question = self.prm.inflate_state(prm_state_question, self.init_number_of_beams)

        input_len = input_ids_question.shape[1]
        complete_beams = defaultdict(list)
        
        proposal_ids, proposal_logits, gen_state = self.generator(input_ids_question, gen_state_question)
        self.trials += 1

        proposal_response_ids = proposal_ids[:, input_len :]
        proposal_response_text = self.generator.tokenizer.batch_decode(proposal_response_ids)
      
        proposal_scores, proposal_score_logits, prm_state = self.compute_step_scores(proposal_response_text, prm_state_question)

        proposal_agg_scores = aggregate(proposal_scores, self.score_aggregation).item()

        is_complete = self.generator.is_complete(proposal_ids)
        if not is_complete[0]:
            complete_beams['CaseType'].append('Candidates')
            complete_beams['answer'] = []
            complete_beams['aggregate_scores'] = []
            complete_beams['step_scores'] = []
            complete_beams['temp'] = [self.generator.temperature]
        else:
            complete_beams['CaseType'].append('Candidates')
            complete_beams['answer'].append(proposal_response_text[0])
            complete_beams['aggregate_scores'].append(proposal_agg_scores)
            complete_beams['step_scores'].append(proposal_scores.tolist())
            complete_beams['temp'].append(self.generator.temperature)

        self.trials = 0
        last_proposal_ids = proposal_ids.clone()
        best_score = proposal_agg_scores
        logger.info(f'[SelfConsistency] Intial {self.trials}/{self.max_trials} : {best_score:.4f}')

        for trial_idx in range(self.max_trials-1):
            self._update_temperature()
            new_proposal_ids, new_proposal_logits, new_gen_state = self.generator(input_ids_question, gen_state_question)
            self.trials += 1

            new_proposal_respose_ids = new_proposal_ids[:, input_len :]

            new_proposal_response_text = self.generator.tokenizer.batch_decode(new_proposal_respose_ids)
            new_proposal_scores, new_proposal_score_logits, prm_state = self.compute_step_scores(new_proposal_response_text, prm_state_question)
            
            new_proposal_agg_scores = aggregate(new_proposal_scores, self.score_aggregation).item()
            logger.info(f'[SelfConsistency] New score {self.trials}/{self.max_trials} : {new_proposal_agg_scores:.4f}')

            complete_beams['CaseType'].append('Candidates')
            complete_beams['answer'].append(new_proposal_response_text[0])
            complete_beams['aggregate_scores'].append(new_proposal_agg_scores)
            complete_beams['step_scores'].append(new_proposal_scores.tolist())
            complete_beams['temp'].append(self.generator.temperature)

        return complete_beams