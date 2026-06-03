"""
3.4  Critique and Evaluation Agent 
==============================================
Task : Evaluate the generated image relative to the input image and
       the refined prompt.

Requirements:
  - Assess: visual relevance, prompt faithfulness, transformation quality
  - Include: CLIP similarity score + human evaluation rubric
  - Produce verdict: "accept" or "revise"

Uses the new `google.genai` SDK (google-genai package).
"""

import os
import textwrap
from pathlib import Path
from typing import Union

import torch
from PIL import Image


class CritiqueEvaluationAgent:
    """
    Critique and Evaluation Agent — computes CLIP cosine similarity
    between the generated image and the refined prompt, and uses Gemini
    to generate a qualitative critique and accept/revise verdict.

    Parameters
    ----------
    api_key : str | None
        Google Gemini API key (for qualitative critique).
    clip_model_id : str
        HuggingFace CLIP model identifier.
    device : str | None
        ``"cuda"`` / ``"cpu"`` / ``None`` (auto-detect).
    """

    CLIP_MODEL = "openai/clip-vit-base-patch32"

    # Rubric criteria and their max scores
    RUBRIC = {
        "Visual Relevance": 5,
        "Prompt Faithfulness": 5,
        "Transformation Quality": 5,
        "Overall Aesthetic": 5,
    }

    def __init__(
        self,
        api_key: Union[str, None] = None,
        clip_model_id: str = CLIP_MODEL,
        device: Union[str, None] = None,
    ):
        self.clip_model_id = clip_model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Gemini optional (for qualitative critique)
        self._genai_client = None
        self._gemini_model = None
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if resolved_key:
            from google import genai
            self._genai_client = genai.Client(api_key=resolved_key)
            # Auto-discover best available model
            from .vision_agent import discover_model
            self._gemini_model = discover_model(self._genai_client)
            print(f"[EvalAgent] Auto-discovered model: {self._gemini_model}")

        # CLIP loaded lazily
        self._clip_model = None
        self._clip_processor = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        input_image_path: Union[str, Path],
        generated_image_path: Union[str, Path],
        refined_prompt: str,
        user_rubric_scores: Union[dict, None] = None,
    ) -> dict:
        """
        Evaluate a generated image.

        Returns
        -------
        dict with keys:
            clip_score, image_to_image_similarity, rubric_scores,
            rubric_total, verdict, critique_text
        """
        input_image_path = Path(input_image_path)
        generated_image_path = Path(generated_image_path)

        self._load_clip()

        print("[EvalAgent] Computing CLIP similarity (prompt <-> generated image) ...")
        clip_score = self._clip_text_image_similarity(
            refined_prompt, generated_image_path
        )

        print("[EvalAgent] Computing image-to-image CLIP similarity ...")
        img_similarity = self._clip_image_image_similarity(
            input_image_path, generated_image_path
        )

        critique_text = self._gemini_critique(
            input_image_path, generated_image_path, refined_prompt,
            clip_score, img_similarity
        )

        rubric_scores = user_rubric_scores or self._estimate_rubric(
            clip_score, img_similarity
        )
        rubric_total = round(
            sum(rubric_scores.values()) / (len(rubric_scores) * 5) * 20, 1
        )

        verdict = "accept" if (clip_score >= 0.20 and rubric_total >= 12.0) else "revise"

        result = {
            "clip_score": round(clip_score, 4),
            "image_to_image_similarity": round(img_similarity, 4),
            "rubric_scores": rubric_scores,
            "rubric_total": rubric_total,
            "verdict": verdict,
            "critique_text": critique_text,
        }

        self._print_summary(result)
        return result

    def print_rubric_table(self, examples: list[dict]) -> None:
        """Print a formatted human evaluation rubric table."""
        criteria = list(self.RUBRIC.keys())
        header = f"{'Example':<22} | " + " | ".join(f"{c:<25}" for c in criteria) + \
                 " | CLIP   | Verdict"
        sep = "-" * len(header)
        print("\n" + "HUMAN EVALUATION RUBRIC".center(len(header)))
        print(sep)
        print(header)
        print(sep)
        for ex in examples:
            name = ex.get("label", "?")[:20]
            scores = ex.get("rubric_scores", {})
            row = f"{name:<22} | "
            for c in criteria:
                s = scores.get(c, "-")
                row += f"{str(s)+'/5':<25} | "
            row += f"{ex.get('clip_score', 0.0):.4f} | {ex.get('verdict', '-')}"
            print(row)
        print(sep + "\n")

    # ------------------------------------------------------------------
    # Private -- CLIP helpers
    # ------------------------------------------------------------------

    def _load_clip(self) -> None:
        if self._clip_model is not None:
            return
        from transformers import CLIPProcessor, CLIPModel
        print(f"[EvalAgent] Loading CLIP model '{self.clip_model_id}' ...")
        self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)
        self._clip_model = CLIPModel.from_pretrained(self.clip_model_id).to(self.device)
        print("[EvalAgent] CLIP ready.")

    def _clip_text_image_similarity(self, text: str, image_path: Path) -> float:
        img = Image.open(image_path).convert("RGB")
        inputs = self._clip_processor(
            text=[text], images=img, return_tensors="pt",
            padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._clip_model(**inputs)
        sim = torch.sigmoid(outputs.logits_per_image / 100).item()
        return float(sim)

    def _clip_image_image_similarity(
        self, img1_path: Path, img2_path: Path
    ) -> float:
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        inputs = self._clip_processor(images=[img1, img2], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._clip_model.get_image_features(**inputs)
        # Newer transformers may return an output object instead of a tensor
        if not isinstance(feats, torch.Tensor):
            feats = getattr(feats, 'image_embeds', None) or feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        sim = (feats[0] @ feats[1]).item()
        return float((sim + 1.0) / 2.0)

    # ------------------------------------------------------------------
    # Private -- Gemini critique
    # ------------------------------------------------------------------

    def _gemini_critique(
        self,
        input_path: Path,
        generated_path: Path,
        prompt: str,
        clip_score: float,
        img_similarity: float,
    ) -> str:
        if self._genai_client is None:
            return (
                "Qualitative critique unavailable (no GOOGLE_API_KEY). "
                f"CLIP prompt-image score: {clip_score:.4f}. "
                f"Image-to-image similarity: {img_similarity:.4f}."
            )
        try:
            input_img = Image.open(input_path)
            gen_img = Image.open(generated_path)
            critique_prompt = textwrap.dedent(f"""
                You are an expert image quality evaluator.

                You are given:
                  - An original/source image (Image 1)
                  - A newly generated image (Image 2) created using this prompt:
                    "{prompt}"

                Automatic scores:
                  - CLIP prompt-faithfulness: {clip_score:.4f} (0-1, higher is better)
                  - Image-to-image similarity: {img_similarity:.4f}

                Write a SHORT critique (4-6 sentences) covering:
                  1. Visual Relevance
                  2. Prompt Faithfulness
                  3. Transformation Quality
                  4. Final verdict - end with either:
                       VERDICT: accept
                     or
                       VERDICT: revise
                     with a one-line reason.
            """).strip()
            import time
            from google.genai import errors as genai_errors
            from .vision_agent import QuotaError
            delay, max_retries = 10, 5
            for attempt in range(max_retries):
                try:
                    response = self._genai_client.models.generate_content(
                        model=self._gemini_model,
                        contents=[critique_prompt, input_img, gen_img],
                    )
                    return response.text
                except genai_errors.ClientError as e:
                    err_msg = str(e)
                    if '429' in err_msg:
                        if 'limit: 0' in err_msg:
                            raise QuotaError(
                                "Gemini API quota is 'limit: 0'. Please link a billing account "
                                "at https://aistudio.google.com/app/settings to activate your quota."
                            ) from e

                        if attempt < max_retries - 1:
                            print(f"  [EvalAgent] Rate limited. Retrying in {delay}s ...")
                            time.sleep(delay)
                            delay *= 2
                        else:
                            raise
                    else:
                        raise
        except Exception as e:
            return f"Critique generation failed: {e}"

    # ------------------------------------------------------------------
    # Private -- heuristic rubric
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_rubric(clip_score: float, img_similarity: float) -> dict:
        def _scale(val: float, lo: float, hi: float) -> int:
            clamped = max(lo, min(hi, val))
            return max(1, min(5, round(1 + 4 * (clamped - lo) / (hi - lo))))

        return {
            "Visual Relevance": _scale(img_similarity, 0.3, 0.9),
            "Prompt Faithfulness": _scale(clip_score, 0.15, 0.35),
            "Transformation Quality": _scale(
                (clip_score + img_similarity) / 2, 0.2, 0.65
            ),
            "Overall Aesthetic": _scale(clip_score, 0.15, 0.35),
        }

    @staticmethod
    def _print_summary(result: dict) -> None:
        print("\n" + "=" * 60)
        print("CRITIQUE & EVALUATION AGENT -- OUTPUT")
        print("=" * 60)
        print(f"CLIP (prompt <-> image) : {result['clip_score']:.4f}")
        print(f"Image-to-image sim     : {result['image_to_image_similarity']:.4f}")
        print(f"\nRubric Scores (each /5):")
        for k, v in result["rubric_scores"].items():
            print(f"  {k:<28}: {v}/5")
        print(f"Rubric Total           : {result['rubric_total']}/20")
        print(f"\nVerdict  : {result['verdict'].upper()}")
        print(f"\nCritique :\n{result['critique_text']}")
        print("=" * 60 + "\n")
