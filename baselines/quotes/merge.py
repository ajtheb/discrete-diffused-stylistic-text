import json



def main():
    # 1. Load mandela JSON data from file
    with open('nelson_mandela_quotes.json', 'r', encoding='utf-8') as f:
        quotes_data = json.load(f)

    mandela_quotes1 = [q['quote'] for q in quotes_data]

    # 1. Load gandhi JSON data from file
    with open('mahatma_gandhi_quotes.json', 'r', encoding='utf-8') as f:
        quotes_data = json.load(f)

    gandhi_quotes1 = [q['quote'] for q in quotes_data]


    # 1. Load quotes JSON data from file
    with open('quotes.json', 'r', encoding='utf-8') as f:
        quotes_data = json.load(f)

    mandela_quotes2 = quotes_data['mandela_quotes']
    gandhi_quotes2 = quotes_data['gandhi_quotes']

    mandela_quotes_merged = list(set(mandela_quotes1 + mandela_quotes2))
    gandhi_quotes_merged = list(set(gandhi_quotes1 + gandhi_quotes2))

    print("Number of mandela quotes: ", len(mandela_quotes_merged))
    print("Number of gandhi quotes: ", len(gandhi_quotes_merged))

    quotes_data = {
        "authors": {
            "Nelson Mandela": {
                "quotes": mandela_quotes_merged,
                "count": len(mandela_quotes_merged)
            },
            "Mahatma Gandhi": {
                "quotes": gandhi_quotes_merged,
                "count": len(gandhi_quotes_merged)
            }
        },
        "total_quotes": len(mandela_quotes_merged) + len(gandhi_quotes_merged)
    }

    # Save to JSON file
    with open("merged_quotes.json", "w", encoding="utf-8") as f:
        json.dump(quotes_data, f, indent=2, ensure_ascii=False)




if __name__ == '__main__':
    main()


