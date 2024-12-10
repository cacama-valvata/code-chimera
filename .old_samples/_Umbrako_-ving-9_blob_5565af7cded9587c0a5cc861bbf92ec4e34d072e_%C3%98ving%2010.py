# -----------------------------------------------------------
#                     Oppgave d
# -----------------------------------------------------------

import json
from datetime import datetime
from typing import List, Optional


class Avtale:
    def __init__(self, tittel: str, sted: str, start: datetime, varighet: int, Kategori):
        self.tittel = tittel
        self.sted = sted
        self.start = start
        self.varighet = varighet
        self.Kategori = self.Kategori()
        
        class Kategori:
            def __init__(self, ID: str, navn: str, prioritet: str):
                self.ID = ID
                self.navn = navn
                self.prioritet = prioritet
                
                
                
            def __str__(self):
             return f"ID={self.ID}, navn={self.navn}, prioritet={self.prioritet}"

    # Oppgave e
    def __str__(self):
        return f"tittel={self.tittel}, sted={self.sted}, start={self.start}, varighet={self.varighet},"


# -----------------------------------------------------------
#                     Oppgave f
# -----------------------------------------------------------

# Kommentaren kan utvides
def lag_ny_avtale() -> Optional[Avtale]:
    """Registrer en ny avtale"""

    tittel = input("Skriv inn avtale tittel # ")
    sted = input("Skriv inn avtale sted # ")
    start = input("Skriv inn avtale start 'ÅÅÅÅ-MM-DD HH:MM:SS' # ")
    varighet = input("Skriv inn avtale varighet (minutter) # ")
 

    # Sjekk om vi kan konvertere start til et datetime objekt
    try:
        start_tid = datetime.fromisoformat(start)
    except ValueError:
        print("Dato-formatet du oppgav ble ikke gjenkjent, bruk 'ÅÅÅÅ-MM-DD HH:MM:SS'")
        return None

    # Sjekk om varighet er et tall
    try:
        varighet_int = int(varighet)
    except ValueError:
        print("Du må skrive inn et gyldig tall for varighet!")
        return None

    return Avtale(tittel, sted, start_tid, varighet_int)


# -----------------------------------------------------------
#                     Oppgave g
# -----------------------------------------------------------
def skriv_ut_avtaler(avtaler: List[Avtale], overskrift=None) -> None:
    """Skriv ut en liste med avtaler til skjermen"""

    # Pga. overskrift er frivillig må vi først se om overskrift ble oppgitt
    if overskrift:
        print(overskrift)

    # Loop gjennom listen og skriv ut indeks + tittel til avtale
    for i in range(len(avtaler)):
        print(f"#{i} {avtaler[i].tittel}")


# -----------------------------------------------------------
#                     Oppgave h
# -----------------------------------------------------------

# Hentet fra https://stackoverflow.com/a/60035604
# og https://stackoverflow.com/a/35780962
# Vi trenger denne for å konvertere datetime object til string
def datetime_option(obj):
    if isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    else:
        return obj.__dict__


def skriv_avtaler_til_fil(avtaler: List[Avtale]) -> None:
    """Skriv liste med avtaler til json fil"""
    json_object = json.dumps(avtaler, default=datetime_option,
                             indent=4, separators=(',', ': '))

    with open("avtale_lister.json", "w") as json_fil:
        json_fil.write(json_object)


# -----------------------------------------------------------
#                     Oppgave i
# -----------------------------------------------------------


def les_avtaler_fra_fil() -> List[Avtale]:
    """Les inn alle avtaler fra fil og returner i liste"""
    avtaler = []
    with open("avtale_lister.json", "r") as json_fil:
        json_data = json.load(json_fil)

        # Det er trygt å anta at typene til
        # variablene er korrekt pga. vi skriver
        # bare gyldig "avtaler" til filen
        for avtale_json in json_data:
            tittel = avtale_json["tittel"]
            sted = avtale_json["sted"]
            start = datetime.fromisoformat(avtale_json["start"])
            varighet = int(avtale_json["varighet"])

            ny_avtale = Avtale(tittel, sted, start, varighet)
            avtaler.append(ny_avtale)

        return avtaler


# -----------------------------------------------------------
#                     Oppgave j
# -----------------------------------------------------------
def alle_avtaler_paa_dato(avtaler: List[Avtale], dato: datetime):
    """Returner alle avtale på bestemt dato"""
    avtaler_paa_dato = []
    for avtale in avtaler:
        if avtale.start == dato:
            avtaler_paa_dato.append(avtale)
    return avtaler_paa_dato


# -----------------------------------------------------------
#                     Oppgave k
# -----------------------------------------------------------

def sok_etter_avtale(avtaler: List[Avtale], soke_ord_liste: List[str]):
    """Søk etter en avtale i en liste"""

    sok_resultat = []
    for avtale in avtaler:

        # Gjør søket ufølsomt mot små/store bokstaver
        avtale_tittel = avtale.tittel.lower()

        # Lag en kopi av soke ordene
        soke_ord_liste_kopi = soke_ord_liste.copy()
        for soke_ord in soke_ord_liste:
            if soke_ord in avtale_tittel:
                soke_ord_liste_kopi.remove(soke_ord)

            # Dersom soke_ord_liste_kopi er tom er søket fullført
            if not soke_ord_liste_kopi:
                sok_resultat.append(avtale)
                continue

    return sok_resultat


# -----------------------------------------------------------
#                     Oppgave l, m, n, o
# -----------------------------------------------------------


def rediger_avtale(avtale: Avtale):
    """Rediger en avtale"""

    valg = input("""[1] Endre tittel
[2] Endre sted
[3] Endre start
[4] Endre varighet
Velg en operasjon # """)

    if valg == "1":
        ny_tittel = input(f"Skriv inn en ny tittel for avtalen '{avtale.tittel}' # ")
        avtale.tittel = ny_tittel

    elif valg == "2":
        ny_sted = input(f"Skriv inn et nytt sted for avtalen '{avtale.tittel}' # ")
        avtale.sted = ny_sted

    elif valg == "3":
        start = input(f"Skriv inn en ny start-tid for avtalen '{avtale.tittel}' 'ÅÅÅÅ-MM-DD HH:MM:SS' # ")

        # Gjør om str til datetime
        try:
            ny_start = datetime.fromisoformat(start)
            avtale.start = ny_start
        except ValueError:
            print("Dato-formatet du oppgav ble ikke gjenkjent, bruk 'ÅÅÅÅ-MM-DD HH:MM:SS'")

    elif valg == "4":
        vargihet = input(f"Skriv inn en ny varighet avtalen '{avtale.tittel}' # ")

        try:
            ny_varighet = int(vargihet)
            avtale.varighet = ny_varighet
        except ValueError:
            print("Du oppga ikke en int!")


def kjor_meny_system():

    mine_avtaler = []
    endret = False

    while True:
        valg = input("""[1] Lese inn avtaler fra fil
[2] Skrive avtalene til fil
[3] Lag ny avtale
[4] Skrive ut alle avtalene
[5] Rediger en avtale
[6] Slett en avtale
[7] Avslutt
Velg en operasjon # """)

        if valg == "1":
            # Spør om bekfreftelse før vi leser inn dersom mine_avtaler ikke er tom
            if mine_avtaler:
                bekreftelse = input("Vil du overskrive avtaler (Ja/Nei) # ").lower()
                if bekreftelse == "ja":
                    mine_avtaler = les_avtaler_fra_fil()
            else:
                mine_avtaler = les_avtaler_fra_fil()

        elif valg == "2":
            skriv_avtaler_til_fil(mine_avtaler)
            endret = False

        elif valg == "3":
            min_nye_avtale = lag_ny_avtale()
            if isinstance(min_nye_avtale, Avtale):
                mine_avtaler.append(min_nye_avtale)
                endret = True

        elif valg == "4":
            overskrift = input("Skriv inn overskrift # ")
            skriv_ut_avtaler(mine_avtaler, overskrift=overskrift)

        elif valg == "5":
            skriv_ut_avtaler(mine_avtaler)

            valgt_indeks = input("Velg indeksen til avtalen du vil redigere # ")
            try:
                valgt_indeks_int = int(valgt_indeks)

                valgt_avtale = mine_avtaler[valgt_indeks_int]
                rediger_avtale(valgt_avtale)
                endret = True
            except ValueError:
                print("Du oppga ikke en int!")

        elif valg == "6":
            skriv_ut_avtaler(mine_avtaler)

            valgt_indeks = input("Velg indeksen til avtalen du vil slette # ")
            try:
                valgt_indeks_int = int(valgt_indeks)

                del mine_avtaler[valgt_indeks_int]
                endret = True
            except ValueError:
                print("Du oppga ikke en int!")

        elif valg == "7":
            if endret:
                lagre_valg = input("Vil du lagre ? (Ja/Nei) ").lower()
                if lagre_valg == "ja":
                    skriv_avtaler_til_fil(mine_avtaler)
            break


if __name__ == '__main__':
    kjor_meny_system()
    
###### ØVING 10 ######    


    
    #Legg inn __STR__ (SE  LINJE 322!))
        
    
def lag_ny_kategori() -> Optional[Avtale.Kategori]:
  

    ID = input("Skriv inn  ID # ")
    navn = input("Skriv inn et Navn # ")
    prioritet = input("Skriv inn en prioritet # ")


    try:
        prioritet_int = int(prioritet)
    except ValueError:
        print("Du må skrive inn et gyldig tall for prioritet!")
        return None

    return Avtale.Kategori(ID, navn, prioritet_int)

def skrive_kategori_til_fil(Kategori: List[Avtale.Kategori]) -> None:
    "Skriver liste med kategorier til fil"
    json_object = json.dumps(Kategori, default=datetime_option,
                             indent=4, seperators=(',',':'))
    with open("kategori_lister.json", "w") as json_fil:
        json_fil.write(json_object)

def skriv_ut_kategori(Kategori: List[Avtale.Kategori], overskrift1=None) -> None:
    "Skriver ut en liste med kategorier til skjermen"
    
    #Pga. Overskrift1 er frivillig så må vi se om overskriften er oppgitt.
    if overskrift1:
        print(overskrift1)
        
    #loop gjennom listen og skrive ut indeks + tittel til avtale
    for i in range(len(Kategori)):
        print(f"#{i} {Kategori[i].tittel}")
        
class Sted:
    def __init__(self,ID: str, navn: str, prioritet: str):
        self.ID = ID
        self.navn= navn
        self.prioritet = prioritet

    def __str__(self):
        return  f"ID ={self.ID}, navn={self.navn}, prioritet={self.prioritet}"

def lag_nytt_sted() -> Optional[Sted]:
    
    ID = input("Skriv inn gateadresse #")
    navn = input("Skriv inn et poststed #")
    prioritet = input("Skriv inn en postnummer")


    
    try:
        prioritet_int = int(prioritet)
    except ValueError:
        print("Du må skrive inn ett positivt tall!")
        return None

    return Sted(ID, navn, prioritet_int)

def skrive_sted_til_fil(sted: List[Sted]) -> None:
    "Skriver liste med steder til filen"
    json_object = json.dumps(sted, default=datetime_option,
                             indent=4, seperators=(',',':'))
    with open("sted_lister.json", "w") as json_fil:
        json_fil.write(json_object)
    
def skrive_ut_sted(sted: List[Sted], overskrift2=None) -> None:
    "skriver ut liste med steder til skjermen"
    
    if overskrift2:
        print(overskrift2)
        
    for i in range(len(Sted)):
        print(f"{i} {sted[i].tittel}")
                    
    



    
    


    






