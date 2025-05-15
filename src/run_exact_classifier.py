from src.nguyen_du_classifier import train_and_save_classifier, predict_authorship

if __name__ == "__main__":
    # Train the model
    classifier = train_and_save_classifier()
    
    # Test some verses
    test_verses = [
        "Trăm năm trong cõi người ta,",  # Famous Nguyễn Du verse
        "Chữ tài chữ mệnh khéo là ghét nhau.",  # Famous Nguyễn Du verse
        "Mặt nhìn mặt càng thêm tươi,",  # From Truyện Kiều
        "Thân em vừa trắng lại vừa tròn",  # Hồ Xuân Hương
    ]
    
    print("\nTesting example verses:")
    for verse in test_verses:
        predict_authorship(verse, classifier)
    
    # Interactive mode
    print("\n=== Nguyễn Du Verse Classifier ===")
    print("Enter a verse to check if it was written by Nguyễn Du (or 'quit' to exit)")
    
    while True:
        verse = input("\n> ")
        if verse.lower() == 'quit':
            break
        
        predict_authorship(verse, classifier)