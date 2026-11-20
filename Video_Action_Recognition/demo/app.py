"""Gradio UI from notebook Cell 17, adapted for local VS Code / Windows.

Run from project root: python -m demo.app
Or directly: python demo/app.py
"""
import argparse
import os
import sys
from pathlib import Path

# Allow VS Code's "Run Python File" as well as `python -m demo.app`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
from demo.inference import ActionPredictor, DEFAULT_CHECKPOINT
from demo.video_utils import extract_video_frames, process_frame


def make_interface(predictor):
    """Create exactly the two original demo tabs: live webcam & upload video."""
    def predict_video(path):
        if not path:
            raise gr.Error("Please upload or record a video")
        try:
            frames = extract_video_frames(path)
            out = predictor.predict_frames(frames)
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc
        result = f"### Predicted Action: {out['action']}\n### Confidence: {out['confidence']*100:.2f}%"
        info = ("Original sampled frames: 16\nModel input frames: 8\n"
                "Resolution: 112x112\n"
                f"Inference Time: {out['inference_ms']:.2f} ms")
        return result, out["scores"], info

    def webcam_predict(frame, state):
        if not state:
            state = {"frames": [], "count": 0, "result": "### Waiting...", "scores": {}}
        if frame is None:
            return state["result"], state["scores"], "Waiting for camera...", state
        processed = process_frame(frame)
        state["frames"].append(processed)
        state["frames"] = state["frames"][-16:]
        state["count"] += 1
        count = len(state["frames"])
        if count < 16:
            return "### Collecting video frames...", {}, f"Collecting Frames: {count}/16", state
        if (state["count"] - 16) % 4 == 0:
            out = predictor.predict_frames(state["frames"])
            state["result"] = (
                f"### Predicted Action: {out['action']}\n"
                f"### Confidence: {out['confidence']*100:.2f}%"
            )
            state["scores"] = out["scores"]
            state["inference_time"] = out["inference_ms"]
        status = ("LIVE CAMERA ACTIVE\nFrames used: 16 -> 8\n"
                  f"Prediction Time: {state.get('inference_time', 0):.2f} ms")
        return state["result"], state["scores"], status, state

    def reset_camera():
        state = {"frames": [], "count": 0, "result": "### Waiting...", "scores": {}}
        return state, "### Waiting...", {}, "Camera reset successfully."

    with gr.Blocks(title="AI Video Action Recognition") as demo:
        gr.Markdown(
            "# AI VIDEO ACTION RECOGNITION\n"
            "### Deep Learning Final Project\n"
            "**Model:** Pretrained MobileNetV3  |  **Classes:** 10  |  "
            "**Input:** 8 RGB Frames (112×112)\n\n"
            "**Reported Test Accuracy:** 93.33%"
        )
        with gr.Tabs():
            with gr.Tab("LIVE WEBCAM"):
                gr.Markdown("### Start your laptop camera\nThe system continuously updates predictions.")
                webcam = gr.Image(label="Laptop Camera", sources=["webcam"],
                                  type="numpy", streaming=True, height=360)
                camera_state = gr.State({
                    "frames": [], "count": 0, "result": "### Waiting...", "scores": {}
                })
                with gr.Row():
                    live_result = gr.Markdown("### Waiting for camera...")
                    live_scores = gr.Label(label="Top 3 Predictions", num_top_classes=3)
                live_status = gr.Textbox(label="Camera Status", interactive=False)
                reset_button = gr.Button("Reset Camera Predictions")
                webcam.stream(
                    fn=webcam_predict, inputs=[webcam, camera_state],
                    outputs=[live_result, live_scores, live_status, camera_state],
                    stream_every=0.4, time_limit=180, concurrency_limit=1,
                )
                reset_button.click(
                    fn=reset_camera, inputs=[],
                    outputs=[camera_state, live_result, live_scores, live_status],
                )
            with gr.Tab("UPLOAD VIDEO"):
                gr.Markdown("### Upload or record a video")
                video_input = gr.Video(label="Select Video", sources=["upload", "webcam"],
                                       format="mp4")
                predict_button = gr.Button("PREDICT ACTION", variant="primary")
                with gr.Row():
                    video_result = gr.Markdown("### Upload a video to start.")
                    video_scores = gr.Label(label="Top 3 Predictions", num_top_classes=3)
                video_status = gr.Textbox(label="Inference Information", interactive=False)
                predict_button.click(
                    fn=predict_video, inputs=[video_input],
                    outputs=[video_result, video_scores, video_status],
                )
        gr.Markdown("---\n**Video Action Recognition — Final Project**  |  "
                    "PyTorch + MobileNetV3 + Gradio\n\n"
                    "Confidence is not a guarantee of correctness.")
    return demo


def main():
    parser = argparse.ArgumentParser(description="Local Video Action Recognition demo")
    parser.add_argument("--checkpoint", default=os.getenv("ACTION_MODEL_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default localhost)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--share", action="store_true", help="Create a TEMPORARY public Gradio link")
    args = parser.parse_args()
    print("Loading trained checkpoint:", args.checkpoint)
    predictor = ActionPredictor(args.checkpoint, args.device)
    print("Device:", predictor.device, "| Classes: 10")
    demo = make_interface(predictor)
    demo.queue().launch(
        server_name=args.host, server_port=args.port,
        share=args.share, inbrowser=True, show_error=True,
    )

if __name__ == "__main__":
    main()
