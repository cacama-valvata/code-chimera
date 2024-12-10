import requests
from bs4 import BeautifulSoup

encabezados = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

url = "https://stackoverflow.com/questions"

respuesta = requests.get(url, headers=encabezados)

soup = BeautifulSoup(respuesta.text, 'lxml')

contenedor_preguntas = soup.find(id="questions")
lista_preguntas = contenedor_preguntas.find_all('div', class_="s-post-summary")
for pregunta in lista_preguntas:
    question_title = pregunta.find('h3').text.strip()
    desc = pregunta.find('div', class_="s-post-summary--content-excerpt")
    desc = desc.text.replace("\n", "").replace("\r", "").strip()

    print(question_title)
    print(desc)
    print()
