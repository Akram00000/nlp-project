"""Test script to verify hadith loader works with GitHub hadith-json format."""

from pathlib import Path
from hadith_loader import HadithLoader

def test_hadith_loader():
    """Test loading hadiths from bukhari.json"""
    
    # Path to hadith data
    hadith_dir = Path(r"d:\_hsproject\nlp project\hadith-json\db\by_book\the_9_books")
    bukhari_file = hadith_dir / "bukhari.json"
    
    # Initialize loader
    loader = HadithLoader()
    
    print("Testing HadithLoader with GitHub hadith-json format...\n")
    print(f"Loading from: {bukhari_file}\n")
    
    # Load first 5 hadiths from Bukhari
    hadiths = []
    for i, hadith in enumerate(loader.load_json(bukhari_file)):
        hadiths.append(hadith)
        if i >= 4:  # First 5
            break
    
    print(f"Successfully loaded {len(hadiths)} hadiths\n")
    
    # Display first hadith
    if hadiths:
        print("=" * 80)
        print("SAMPLE HADITH:")
        print("=" * 80)
        h = hadiths[0]
        print(f"ID: {h.id}")
        print(f"Source: {h.source}")
        print(f"Number: {h.number}")
        print(f"Narrator: {h.narrator}")
        print(f"\nArabic Text (first 200 chars):")
        print(h.content[:200] if len(h.content) > 200 else h.content)
        print(f"\nMetadata:")
        for key, value in h.metadata.items():
            if key != 'english_text':  # Skip long English text
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: [English translation available]")
        print("=" * 80)
    
    # Get statistics from all 9 books
    print("\nLoading from all 9 books...")
    all_hadiths = []
    for hadith in loader.load_directory(hadith_dir):
        all_hadiths.append(hadith)
        if len(all_hadiths) >= 100:  # Just load first 100 for quick test
            break
    
    print(f"\nLoaded {len(all_hadiths)} hadiths from directory")
    
    # Display statistics
    stats = loader.get_statistics(all_hadiths)
    print("\nSTATISTICS:")
    print(f"Total hadiths: {stats.get('total_hadiths', 0)}")
    print(f"Unique sources: {stats.get('unique_sources', 0)}")
    print("\nTop sources:")
    for source, count in stats.get('top_sources', []):
        print(f"  {source}: {count} hadiths")
    print("\nTop narrators:")
    for narrator, count in stats.get('top_narrators', [])[:5]:
        print(f"  {narrator}: {count} hadiths")

if __name__ == "__main__":
    test_hadith_loader()
