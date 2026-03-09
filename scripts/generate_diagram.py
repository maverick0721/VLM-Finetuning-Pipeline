from graphviz import Digraph
import os


OUTPUT_DIR = "reports"


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dot = Digraph()

    dot.node("A", "Dataset")
    dot.node("B", "Download Dataset")
    dot.node("C", "Prepare Dataset")

    dot.node("D", "QLoRA Training")
    dot.node("E", "Unsloth Training")

    dot.node("F", "Evaluation")
    dot.node("G", "Benchmark")
    dot.node("H", "Report Generation")
    dot.node("I", "Demo UI")

    dot.edge("A", "B")
    dot.edge("B", "C")

    dot.edge("C", "D")
    dot.edge("C", "E")

    dot.edge("D", "F")
    dot.edge("E", "F")

    dot.edge("F", "G")
    dot.edge("G", "H")
    dot.edge("H", "I")

    path = f"{OUTPUT_DIR}/pipeline_diagram"

    dot.render(path, format="png", cleanup=True)

    print("Pipeline diagram saved:", path + ".png")


if __name__ == "__main__":
    main()