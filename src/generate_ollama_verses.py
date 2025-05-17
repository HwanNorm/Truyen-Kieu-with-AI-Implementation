import argparse
from src.ollama_kieu_generator import OllamaKieuGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate Vietnamese verses using Ollama LLaMA models')
    
    # Basic parameters
    parser.add_argument('--data', type=str, default='data/truyen_kieu.txt',
                        help='Path to Truyện Kiều text file')
    parser.add_argument('--model', type=str, default='llama3',
                        help='Ollama model name (default: llama3)')
    
    # Generation options
    parser.add_argument('--mode', type=str, 
                        choices=['verse', 'pair', 'interactive'],
                        default='verse',
                        help='Generation mode: single verse, verse pair, or interactive')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Initial phrase for verse generation')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of verses to generate')
    parser.add_argument('--max-length', type=int, default=100,
                        help='Maximum length for generation')
    parser.add_argument('--save', action='store_true',
                        help='Save generated verses to file')
    parser.add_argument('--output-dir', type=str, default='output/',
                        help='Directory for output files')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Initialize generator
        print(f"Initializing OllamaKieuGenerator with model {args.model}...")
        generator = OllamaKieuGenerator(
            model_name=args.model,
            truyen_kieu_path=args.data
        )
        
        if args.mode == 'verse':
            print(f"Generating {args.num_samples} verses" + 
                  (f" starting with '{args.prompt}'" if args.prompt else ""))
            
            verses = generator.generate_verse(
                initial_phrase=args.prompt,
                num_samples=args.num_samples,
                max_length=args.max_length
            )
            
            for i, verse in enumerate(verses, 1):
                print(f"\n=== Generated Verse {i} ===")
                print(verse)
                
                # Save if requested
                if args.save:
                    import os
                    os.makedirs(args.output_dir, exist_ok=True)
                    output_file = os.path.join(args.output_dir, f"ollama_verse_{i}.txt")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(verse)
                    print(f"Verse saved to {output_file}")
        
        elif args.mode == 'pair':
            print(f"Generating verse pair" + 
                  (f" starting with '{args.prompt}'" if args.prompt else ""))
            
            luc_verse, bat_verse = generator.generate_verse(
                initial_phrase=args.prompt
            )
            
            print("\n=== Generated Verse Pair ===")
            print(f"Lục verse: {luc_verse}")
            print(f"Bát verse: {bat_verse}")
            
            # Save if requested
            if args.save:
                import os
                os.makedirs(args.output_dir, exist_ok=True)
                output_file = os.path.join(args.output_dir, "ollama_verse_pair.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"{luc_verse}\n{bat_verse}")
                print(f"Verse pair saved to {output_file}")
        
        elif args.mode == 'interactive':
            print("\n=== Ollama Truyện Kiều Verse Generator (Interactive Mode) ===")
            print("Enter a starting phrase (or 'quit' to exit):")
            
            while True:
                prompt = input("> ")
                if prompt.lower() == 'quit':
                    break
                
                verses = generator.generate_verse(
                    initial_phrase=prompt,
                    num_samples=1
                )
                
                if verses:
                    print("\nGenerated verse:")
                    print(verses[0])
                else:
                    print("Failed to generate verse. Try a different prompt.")
                print()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main()