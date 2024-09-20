def return_shots(id, player):
    response = requests.get(
        f"https://www.sofascore.com/football/match/brentford-manchester-city/rsab#id:{id}",
        headers={"User-Agent": "Mozilla/5.0"}
    )

    soup = BeautifulSoup(response.text, "html.parser")
    soup.select('g[cursor="pointer"]')

    headers = {
        'authority': 'api.sofascore.com',
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'max-age=0',
        'dnt': '1',
        'if-none-match': 'W/"4bebed6144"',
        'origin': 'https://www.sofascore.com',
        'referer': 'https://www.sofascore.com/',
        'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    }

    response = requests.get(f'https://api.sofascore.com/api/v1/event/{id}/shotmap', headers=headers)
    shots = response.json()

    return {
        "shotmap": [
            shot for shot in shots["shotmap"]
            if shot["player"]["name"] == player
        ]
    }