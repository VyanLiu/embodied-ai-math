from code_demo.image_search_engine import ImageSearchEngine
from code_demo.search_strategies import CLIPSearchStrategy
from PIL import Image

MODEL_NAME = "../model_saved/clip-vit-base-patch32"
# Adjust paths if running from project root or code_demo folder
IMAGE_PATH_ROOT = "../../../DataSet/search_images"

if __name__ == "__main__":
    # 1. Choose the Strategy (e.g., CLIP)
    print("Initializing CLIP Strategy...")
    clip_strategy = CLIPSearchStrategy(MODEL_NAME)

    # 2. Inject Strategy into Context (Engine)
    engine = ImageSearchEngine(strategy=clip_strategy)

    # 3. Load Data
    try:
        engine.load_images(IMAGE_PATH_ROOT)
    except ValueError as e:
        print(e)
        exit(1)

    # 4. Main Interaction Loop
    while True:
        f_continue, query_text = engine.get_query_text_from_console()
        
        if not f_continue:
            break

        results = engine.search(query_text, top_k=5)

        print("\n--- Top Results ---")
        for filename, score, image in results:
            print(f"Image: {filename:<30} | Score: {score:.4f} ({score*100:.2f}%)")
        print("-------------------\n")

        for filename, score, image in results:
            image.show()
    print("Goodbye!")