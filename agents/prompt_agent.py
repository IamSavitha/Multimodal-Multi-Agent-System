"""
3.2  Prompt Engineering Agent 
=========================================
Task : Refine the user's instruction into a clearer, richer prompt
       suitable for image generation.

Requirements:
  - Use the user request AND the Vision Understanding Agent output to:
      • rewrite the prompt clearly
      • enrich it with visual details from the image analysis
      • preserve the intended transformation

Uses the new `google.genai` SDK (google-genai package).
"""

import os
import textwrap
from typing import Union

from google import genai


class PromptEngineeringAgent:
    """
    Prompt Engineering Agent — uses Gemini to rewrite and enrich the
    user's generation instruction based on visual context from the
    Vision Understanding Agent.

    Parameters
    ----------
    api_key : str | None
        Google Gemini API key.  Falls back to GOOGLE_API_KEY env var.
    model_name : str | None
        Gemini text model to use.  When *None* (default), the best
        available Flash model is auto-discovered from the API.
    """

    def __init__(
        self,
        api_key: Union[str, None] = None,
        model_name: Union[str, None] = None,
    ):
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not resolved_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY not set.  Pass api_key= or set the "
                "environment variable before instantiating PromptEngineeringAgent."
            )
        self._client = genai.Client(api_key=resolved_key)

        if model_name is None:
            from .vision_agent import discover_model
            self.model_name = discover_model(self._client)
            print(f"[PromptAgent] Auto-discovered model: {self.model_name}")
        else:
            self.model_name = model_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine(
        self,
        user_instruction: str,
        vision_output: dict,
    ) -> dict:
        """
        Refine *user_instruction* using visual context from a Vision Agent
        output dict.

        Parameters
        ----------
        user_instruction : str
            Raw instruction from the user.
        vision_output : dict
            Dict returned by ``VisionUnderstandingAgent.analyze()``.

        Returns
        -------
        dict with keys:
            original_instruction, refined_prompt, enrichment_notes,
            transformation_type, model_used, raw_response
        """
        prompt = self._build_prompt(user_instruction, vision_output)

        print(f"[PromptAgent] Refining user instruction with {self.model_name} ...")
        response = self._call_with_retry(prompt)
        raw = response.text

        result = self._parse_response(raw, user_instruction)
        result["model_used"] = self.model_name
        result["raw_response"] = raw

        self._print_summary(result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_with_retry(self, contents, max_retries: int = 5):
        """Call generate_content with exponential backoff on 429 errors."""
        import time
        from google.genai import errors as genai_errors
        from .vision_agent import QuotaError
        delay = 10
        for attempt in range(max_retries):
            try:
                return self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )
            except genai_errors.ClientError as e:
                err_msg = str(e)
                if '429' in err_msg:
                    if 'limit: 0' in err_msg:
                        raise QuotaError(
                            "Gemini API quota is 'limit: 0'. Please link a billing account "
                            "at https://aistudio.google.com/app/settings to activate your quota."
                        ) from e

                    if attempt < max_retries - 1:
                        print(f"  [PromptAgent] Rate limited (429). Retrying in {delay}s ... (attempt {attempt+1}/{max_retries})")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        raise
                else:
                    raise

    def _build_prompt(self, user_instruction: str, vision: dict) -> str:
        objects_str = ", ".join(vision.get("objects", [])) or "N/A"
        vqa_str = "\n".join(
            f"  Q: {item['question']}\n  A: {item['answer']}"
            for item in vision.get("vqa", [])
        )
        return textwrap.dedent(f"""
            You are an expert prompt engineer for image generation models
            (e.g. Stable Diffusion, DALL-E, Flux).

            You have been given:
            1. A user instruction describing the desired image transformation.
            2. Visual context extracted from the source image by a vision model.

            -- USER INSTRUCTION --
            {user_instruction}

            -- VISUAL CONTEXT FROM SOURCE IMAGE --
            Caption          : {vision.get('caption', 'N/A')}
            Key Objects      : {objects_str}
            Scene Description: {vision.get('scene_description', 'N/A')}
            Visual Q&A       :
            {vqa_str}

            Your task: produce an IMPROVED prompt for an image generation
            model that:
              - clearly states what should appear in the output image
              - incorporates key visual details from the source
              - preserves the intended transformation type
              - uses descriptive, generation-friendly language

            Respond in EXACTLY this format:

            ## REFINED PROMPT
            <The complete improved prompt (30-80 words).>

            ## ENRICHMENT NOTES
            <2-3 bullet points explaining what details you added.>

            ## TRANSFORMATION TYPE
            <One of: stylization | variation | enhancement | scene_edit | other>

            Do NOT add extra sections or commentary outside this format.
        """).strip()

    def _parse_response(self, raw: str, original: str) -> dict:
        def _get(text: str, header: str) -> str:
            marker = f"## {header}"
            start = text.find(marker)
            if start == -1:
                return ""
            start += len(marker)
            end = text.find("##", start)
            return text[start:end].strip() if end != -1 else text[start:].strip()

        refined = _get(raw, "REFINED PROMPT")
        notes_raw = _get(raw, "ENRICHMENT NOTES")
        t_type = _get(raw, "TRANSFORMATION TYPE").lower().strip()

        enrichment_notes = [
            line.lstrip("*- ").strip()
            for line in notes_raw.splitlines()
            if line.strip()
        ]

        return {
            "original_instruction": original,
            "refined_prompt": refined,
            "enrichment_notes": enrichment_notes,
            "transformation_type": t_type or "enhancement",
        }

    @staticmethod
    def _print_summary(result: dict) -> None:
        print("\n" + "=" * 60)
        print("PROMPT ENGINEERING AGENT -- OUTPUT")
        print("=" * 60)
        print(f"Original : {result['original_instruction']}")
        print(f"\nRefined Prompt:\n  {result['refined_prompt']}")
        print(f"\nTransformation Type: {result['transformation_type']}")
        print("\nEnrichment Notes:")
        for note in result["enrichment_notes"]:
            print(f"  - {note}")
        print("=" * 60 + "\n")
