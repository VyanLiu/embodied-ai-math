import os
from PIL import Image
from typing import List, Tuple, Optional

# Import only the abstract Strategy interface
try:
    from code_demo.search_strategies import SearchStrategy
except ImportError:
    from search_strategies import SearchStrategy

class ImageSearchEngine:
    def __init__(self, strategy: SearchStrategy):
        """
        Initializes the engine with a specific search strategy.
        Dependency Injection allows us to swap logic easily.
        """
        if not strategy:
            raise ValueError("A valid SearchStrategy must be provided.")

        self.strategy = strategy
        self.candidate_images: List[Image.Image] = []
        self.image_filenames: List[str] = []

    def get_query_text_from_console(self) -> Tuple[bool, str]:
        """
        Prompts the user for input via the console.
        Returns (True, query_text) to continue, or (False, "") to exit.
        """
        while True:
            query_text = input("\nEnter search query (or type 'exit' to quit): ")
        
            if query_text.lower() in ['exit', 'quit']:
                print("Exiting search engine.")
                return False, ""
        
            if not query_text.strip():
                continue # Skip empty inputs
            
            return True, query_text

    def load_images(self, image_root: str):
        """
        Scans the provided directory and loads all valid images into memory.
        """
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

        if not os.path.exists(image_root):
            raise ValueError(f"Directory not found: {image_root}")

        print(f"Loading images from {image_root}...")
    
        count = 0
        for filename in sorted(os.listdir(image_root)):
            if filename.lower().endswith(valid_extensions):
                file_path = os.path.join(image_root, filename)
                try:
                    img = Image.open(file_path).convert("RGB")
                    self.candidate_images.append(img)
                    self.image_filenames.append(filename)
                    count += 1
                except Exception as e:
                    print(f"Skipping {filename}: Could not load image. Error: {e}")
    
        print(f"Successfully loaded {count} images.")

    def search(self, query_text: str, top_k: int = 5, threshold: float = 0.2) -> List[Tuple[str, float, Image.Image]]:
        """
        Searches the loaded images using the injected strategy.
        """
        if not self.candidate_images:
            print("Warning: No images loaded. Use load_images() first.")
            return []

        print(f"Searching for: '{query_text}' among {len(self.candidate_images)} images...")

        # DELEGATION: The engine asks the strategy to do the work.
        probs = self.strategy.predict(query_text, self.candidate_images)

        # Pair results with filenames and images
        results = []
        for i, prob in enumerate(probs):
            score = prob.item()
            if score >= threshold:
                results.append((self.image_filenames[i], score, self.candidate_images[i]))

        # Sort by score descending (the highest probability first)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]