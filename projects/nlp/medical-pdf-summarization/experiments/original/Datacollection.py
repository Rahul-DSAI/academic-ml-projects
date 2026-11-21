# Example of downloading data from a website (e.g., PubMed)
import requests

def download_medical_articles(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Process the data (save to file, etc.)
        return data
    else:
        print("Failed to download data")

# Example usage
url = "https://example.org/medical_articles"
medical_articles = download_medical_articles(url)
