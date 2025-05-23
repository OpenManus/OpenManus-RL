import requests
from bs4 import BeautifulSoup


if __name__ == '__main__':
    f = open("data.csv", "a")
    for tag in ["linux", "bash", "operating-system", "ubuntu"]:
        for i in range(1, 10):
            r = requests.get(f"https://stackoverflow.com/questions/tagged/{tag}?tab=votes&page={i}&pagesize=50")
            soup = BeautifulSoup(r.text, "html.parser")
            question_div = soup.find("div", {"id": "questions"})
            for question in question_div.find_all("div", class_="s-post-summary--content"):
                title = question.h3.a
                f.write(title.string + ", " + title["href"] + "\n")
                f.flush()
    f.close()

