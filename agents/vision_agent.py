"""
3.1  Vision Understanding Agent  
==========================================
Task : Analyze an input image and extract key visual information.

Requirements:
  - Use a vision-language model (Google Gemini Flash) to generate:
      • a short image caption
      • a list of key objects or entities
      • a brief scene description
  - Answer at least two visual questions about the image.

Uses the new `google.genai` SDK (google-genai package).
"""

import os
import textwrap
from pathlib import Path
from typing import Union

from google import genai
from google.genai import types
from PIL import Image


class QuotaError(Exception):
    """Raised when API quota is explicitly zero/blocked."""
    pass


def discover_model(client: genai.Client) -> str:
    """Return a known-good Gemini Flash model for generateContent.

    Note: Auto-discovery via client.models.list() is unreliable because
    the API returns deprecated/versioned models that may no longer be
    available.  We use 'gemini-2.0-flash' which is stable, multimodal,
    and available on all paid accounts.
    """
    return "gemini-2.5-flash"


class VisionUnderstandingAgent:
    """
    Vision Understanding Agent — uses Google Gemini Flash multimodal
    API to analyse an image and return structured visual information.

    Parameters
    ----------
    api_key : str | None
        Google Gemini API key.  Falls back to the ``GOOGLE_API_KEY``
        environment variable when *None*.
    model_name : str | None
        Gemini model identifier.  When *None* (default), the best
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
                "environment variable before instantiating VisionUnderstandingAgent."
            )
        self._client = genai.Client(api_key=resolved_key)

        if model_name is None:
            self.model_name = discover_model(self._client)
            print(f"[VisionAgent] Auto-discovered model: {self.model_name}")
        else:
            self.model_name = model_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        image_path: Union[str, Path],
        visual_questions: list[str] | None = None,
    ) -> dict:
        """
        Analyse *image_path* and return a structured result dict.

        Parameters
        ----------
        image_path : str | Path
            Path to the input image file (JPEG / PNG / WEBP).
        visual_questions : list[str] | None
            Custom visual questions to answer.  Defaults to two generic
            questions when *None*.

        Returns
        -------
        dict with keys:
            caption, objects, scene_description, vqa (list of {q, a}),
            raw_response, image_path, model_used
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if visual_questions is None:
            visual_questions = [
                "What is the dominant color palette in the image?",
                "What mood or atmosphere does this image convey?",
            ]

        # Load image as PIL for the new SDK
        pil_img = Image.open(image_path)
        prompt = self._build_prompt(visual_questions)

        print(f"[VisionAgent] Analysing '{image_path.name}' with {self.model_name} ...")
        response = self._call_with_retry([prompt, pil_img])
        raw = response.text

        result = self._parse_response(raw, visual_questions)
        result["image_path"] = str(image_path)
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
                    # Check for "limit: 0" which specifically means billing/project setup issue
                    if 'limit: 0' in err_msg:
                        raise QuotaError(
                            "Gemini API quota is 'limit: 0'. This usually means your project is NOT "
                            "linked to a billing account in Google AI Studio. Even for the free tier, "
                            "Google now requires a billing linkage to activate the quota.\n\n"
                            "FIX: Link a billing account at https://aistudio.google.com/app/settings"
                        ) from e

                    if attempt < max_retries - 1:
                        print(f"  [VisionAgent] Rate limited (429). Retrying in {delay}s ... (attempt {attempt+1}/{max_retries})")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        raise
                else:
                    raise

    def _build_prompt(self, questions: list[str]) -> str:
        q_block = "\n".join(
            f"   Q{i+1}: {q}" for i, q in enumerate(questions)
        )
        return textwrap.dedent(f"""
            You are a vision analysis assistant.  Carefully examine the provided
            image and respond in the EXACT structured format below.

            ## CAPTION
            <Write a single descriptive sentence (max 25 words) that captures
            what the image shows.>

            ## OBJECTS
            <Comma-separated list of the key objects / entities visible in the
            image.  List at least five.>

            ## SCENE DESCRIPTION
            <Write 2-3 sentences describing the scene, setting, and any
            notable visual features.>

            ## VISUAL Q&A
            Answer each question based ONLY on what is visible in the image.

            {q_block}

            Answer Q1: <answer>
            Answer Q2: <answer>

            Do NOT add extra sections or commentary outside this format.
        """).strip()

    def _parse_response(self, raw: str, questions: list[str]) -> dict:
        """Leniently parse the structured Gemini response."""

        def _extract_section(text: str, header: str) -> str:
            marker = f"## {header}"
            start = text.find(marker)
            if start == -1:
                return ""
            start += len(marker)
            end = text.find("##", start)
            return text[start:end].strip() if end != -1 else text[start:].strip()

        caption = _extract_section(raw, "CAPTION")
        objects_raw = _extract_section(raw, "OBJECTS")
        scene = _extract_section(raw, "SCENE DESCRIPTION")
        vqa_block = _extract_section(raw, "VISUAL Q&A")

        objects = [o.strip() for o in objects_raw.split(",") if o.strip()]

        vqa = []
        for i, q in enumerate(questions):
            tag = f"Answer Q{i+1}:"
            idx = vqa_block.find(tag)
            if idx != -1:
                ans_start = idx + len(tag)
                next_idx = vqa_block.find("Answer Q", ans_start)
                answer = (
                    vqa_block[ans_start:next_idx].strip()
                    if next_idx != -1
                    else vqa_block[ans_start:].strip()
                )
            else:
                answer = "N/A"
            vqa.append({"question": q, "answer": answer})

        return {
            "caption": caption,
            "objects": objects,
            "scene_description": scene,
            "vqa": vqa,
        }

    @staticmethod
    def _print_summary(result: dict) -> None:
        print("\n" + "=" * 60)
        print("VISION UNDERSTANDING AGENT -- OUTPUT")
        print("=" * 60)
        print(f"Caption  : {result['caption']}")
        print(f"Objects  : {', '.join(result['objects'])}")
        print(f"Scene    : {result['scene_description']}")
        print("\nVisual Q&A:")
        for item in result["vqa"]:
            print(f"  Q: {item['question']}")
            print(f"  A: {item['answer']}")
        print("=" * 60 + "\n")
