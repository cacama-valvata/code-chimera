import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import smtplib

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
#print(voices[1].id)
engine.setProperty('voice', voices[0].id)
def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def WishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("good morning and have a great day ")

    elif hour>=12 and hour<18:
        speak("good afternoon!")

    else:
        speak("good evening!")

    speak(" I am Jarvis sir. Please tell me how may i help you")

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening.....")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing....")
        query = r.recognize_google(audio,Language='en-in')
        print(f"User said:{ query}\n")

    except Exception as e:
        #print(e)

        print("Say that again please....")
        return "None"
    return query

#def sendEmail(to, content):
 #   server = smtplib.SMTP('smtp.gmail.com',587)
  #  server.ehlo()
   # server.starttls()
    #server.login('youremail@gmail.com','password')
    #server.sendmail('youremail@gmail.com', to,content)
    #server.close()
    
if __name__ == "__main__":
    speak(" hey rutuja !! whats up!! ")
    WishMe()
    #while True:
    if 1:

        query = takeCommand().lower()
        if 'wikipedia' in query:
            speak('Serching wikipedia...')
            query = query.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences=2)
            speak("According to wikipedia")
            print(results)
            speak(results)

        elif 'open youtube' in query:
            webbrowser.open("youtube.com")

        elif 'open google' in query:
            webbrowser.open("google.com")

        elif 'open stackoverflow' in query:
            webbrowser.open("stackoverflow.com")

    #elif 'play music' in query:
        elif 'the time' in query:
            strTime = datetime.datetime.now().strTime("%H:%M:%S")
            print(strTime)
            speak(f"The time is {strTime}")

        elif 'open code' in query:
            codePath = "C:\\Users\\rutuj\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu"
            os.startfile(codePath)

       # elif 'email to rutuja' in query:
        #    try:
         #       speak("what should i say?")
          #      content = takeCommand()
           #     to = "youremail@gmail.com" 
            #    sendEmail(to, content)
             #   speak("Email has been sent!")
            #except Exception as e:
             #   print(e)
              #  speak("sorry my friend rutuja..i am not able to this email!!")

