"""
3.5  Integration and Execution 
==========================================
Orchestrator — runs the full Multi-Agent Pipeline:

  Vision Understanding  →  Prompt Engineering  →
  Image Generation      →  Critique / Evaluation

Each agent is called independently.  Intermediate outputs are collected
and the final result dict captures the entire trace of the run.

Usage
-----
    from orchestrator import MultiAgentOrchestrator

    orch = MultiAgentOrchestrator(api_key="AIza...")
    result = orch.run(
        image_path="test_images/use_case_1.jpg",
        user_instruction="Transform this into a watercolour painting",
        mode="stylization",
    )
"""

import traceback
from pathlib import Path
from typing import Union

from agents.vision_agent import VisionUnderstandingAgent
from agents.prompt_agent import PromptEngineeringAgent
from agents.generation_agent import ImageGenerationAgent
from agents.evaluation_agent import CritiqueEvaluationAgent


class MultiAgentOrchestrator:
    """
    Orchestrates the four specialized agents in the correct sequence.

    Parameters
    ----------
    api_key : str | None
        Google Gemini API key (used by Vision, Prompt, and Eval agents).
    output_dir : str | Path
        Directory for generated images.
    gen_steps : int
        SD inference steps (trade-off: speed vs quality).
    device : str | None
        ``"cuda"`` / ``"cpu"`` / auto.
    """

    def __init__(
        self,
        api_key: Union[str, None] = None,
        output_dir: Union[str, Path] = "outputs",
        gen_steps: int = 25,
        device: Union[str, None] = None,
    ):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.gen_steps = gen_steps
        self.device = device

        # Instantiate agents
        self.vision_agent = VisionUnderstandingAgent(api_key=api_key)
        self.prompt_agent = PromptEngineeringAgent(api_key=api_key)
        self.gen_agent = ImageGenerationAgent(
            output_dir=output_dir,
            num_inference_steps=gen_steps,
            device=device,
        )
        self.eval_agent = CritiqueEvaluationAgent(
            api_key=api_key,
            device=device,
        )

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        image_path: Union[str, Path],
        user_instruction: str,
        mode: str = "stylization",
        visual_questions: Union[list[str], None] = None,
        seed: Union[int, None] = None,
    ) -> dict:
        """
        Run the full pipeline on a single use case.

        Parameters
        ----------
        image_path : str | Path
            Source image for the pipeline.
        user_instruction : str
            What the user wants to do with the image.
        mode : str
            Generation mode: ``stylization`` / ``variation`` /
            ``enhancement``.
        visual_questions : list[str] | None
            Optional custom VQA questions for the Vision Agent.
        seed : int | None
            Generation seed (for reproducibility).

        Returns
        -------
        dict — full trace:
            image_path, user_instruction, vision_output, prompt_output,
            generation_output, evaluation_output, success, error
        """
        result: dict = {
            "image_path": str(image_path),
            "user_instruction": user_instruction,
            "mode": mode,
            "success": False,
            "error": None,
            "vision_output": None,
            "prompt_output": None,
            "generation_output": None,
            "evaluation_output": None,
        }

        try:
            banner = f"  PIPELINE RUN : {Path(image_path).name}  "
            print("\n" + ">" * 60)
            print(f">{banner.center(58)}>")
            print(">" * 60 + "\n")

            # ── Step 1 : Vision Understanding ──────────────────────────
            print("── STEP 1/4  Vision Understanding ──────────────────────")
            vision_out = self.vision_agent.analyze(
                image_path=image_path,
                visual_questions=visual_questions,
            )
            result["vision_output"] = vision_out

            # ── Step 2 : Prompt Engineering ────────────────────────────
            print("── STEP 2/4  Prompt Engineering ────────────────────────")
            prompt_out = self.prompt_agent.refine(
                user_instruction=user_instruction,
                vision_output=vision_out,
            )
            result["prompt_output"] = prompt_out

            # ── Step 3 : Image Generation ──────────────────────────────
            print("── STEP 3/4  Image Generation ──────────────────────────")
            gen_out = self.gen_agent.generate(
                refined_prompt=prompt_out["refined_prompt"],
                mode=mode,
                seed=seed,
            )
            result["generation_output"] = gen_out

            # ── Step 4 : Critique & Evaluation ────────────────────────
            print("── STEP 4/4  Critique & Evaluation ────────────────────")
            if gen_out["output_paths"]:
                eval_out = self.eval_agent.evaluate(
                    input_image_path=image_path,
                    generated_image_path=gen_out["output_paths"][0],
                    refined_prompt=prompt_out["refined_prompt"],
                )
                result["evaluation_output"] = eval_out
            else:
                result["evaluation_output"] = {"error": "No generated image to evaluate."}

            result["success"] = True
            print("  Pipeline completed successfully.\n")

        except Exception as exc:
            result["error"] = str(exc)
            result["success"] = False
            print(f"\n  Pipeline failed: {exc}")
            traceback.print_exc()
            self._handle_failure(exc, image_path, user_instruction)

        return result

    def run_batch(self, use_cases: list[dict]) -> list[dict]:
        """
        Run the pipeline on multiple use cases.

        Parameters
        ----------
        use_cases : list[dict]
            Each dict: {image_path, user_instruction, mode (opt),
                        visual_questions (opt), seed (opt)}

        Returns
        -------
        list[dict] — one result dict per use case.
        """
        results = []
        for i, uc in enumerate(use_cases):
            print(f"\n{'═'*60}")
            print(f"  USE CASE {i+1}/{len(use_cases)}")
            print(f"{'═'*60}")
            res = self.run(
                image_path=uc["image_path"],
                user_instruction=uc["user_instruction"],
                mode=uc.get("mode", "stylization"),
                visual_questions=uc.get("visual_questions"),
                seed=uc.get("seed"),
            )
            results.append(res)
        return results

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_failure(exc: Exception, image_path, instruction: str) -> None:
        """Gracefully handle and log failures."""
        err_type = type(exc).__name__
        print("\n[Orchestrator] Failure Analysis:")
        if "FileNotFoundError" in err_type:
            print(f"  → Image not found: {image_path}")
            print("    Check that the path is correct and the file exists.")
        elif "EnvironmentError" in err_type or "APIError" in err_type:
            print("  → API error — check GOOGLE_API_KEY is valid and has quota.")
        elif "CUDA" in str(exc) or "out of memory" in str(exc).lower():
            print("  → GPU OOM — try reducing gen_steps or switching to CPU.")
        elif "ambiguous" in instruction.lower() or len(instruction.strip()) < 5:
            print("  → Ambiguous/short instruction — provide a more descriptive prompt.")
        else:
            print(f"  → Unhandled error ({err_type}): {exc}")
