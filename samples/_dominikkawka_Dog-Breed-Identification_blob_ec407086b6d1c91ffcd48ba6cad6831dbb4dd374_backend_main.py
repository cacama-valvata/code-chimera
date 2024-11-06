from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer

import shutil
import re
import os
import datetime

from backend import loadRunModel
from backend import database
from backend import model
from backend import commonVariables as val

app = FastAPI()

#origins = ['https://localhost:3000', 'http://localhost:3000', 'http://localhost:5173', 'https://localhost:5173']

app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"], #origins,
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"]
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/")
async def root():
   return {"message": "hello world"}

@app.post("/uploadImage")
async def uploadImage(image: UploadFile = File(...)):
   lowercase = (image.filename).lower()
   if ('jpeg' in lowercase or 'jpg' in lowercase or 'png' in lowercase):
      #save file to disc
      with open(""+image.filename, "wb") as buffer:
         shutil.copyfileobj(image.file, buffer)

      result = loadRunModel.modelPrediction(image.filename)
      database.save_prediction(result)
      os.remove(image.filename)
      return result
   else:
      raise HTTPException(status_code=400, detail="wrong file extension, please use png or jpg")
   
@app.post("/webcamImage")
async def readWebcamFeed(base64String: model.webcamImage):
   #https://stackoverflow.com/questions/59929028/python-fastapi-error-422-with-post-request-when-sending-json-data
   print(base64String.imageName)
   result = loadRunModel.webcamBase64toJPG(base64String.image, base64String.imageName)
   database.save_prediction(result)
   print(result)
   return result

@app.get("/allPredictions")
async def allPredictions(): 
   response = await database.fetch_all_predictions()
   return response

@app.get("/getPrediction")
async def getPrediction(image):
   response = await database.fetch_prediction_by_image_name(image)
   return response

@app.get("/getDescription")
async def fetch_dog_breed_description(dogBreed):
   response = await database.fetch_dog_breed_description(dogBreed)
   return response

@app.patch("/prediction")
async def patchCorrectBreed(correctBreed: model.submitFeedbackPrediction):
   #https://fastapi.tiangolo.com/tutorial/encoder/
   response = await database.update_breed(correctBreed.predictedBreed, correctBreed.image, correctBreed.actualBreed)
   return response

@app.patch("/predictionUser")
async def patchUserToPrediction(addUserToPred: model.addUserToPrediction):
   response = await database.patch_user_to_prediction(addUserToPred.predictedBreed, addUserToPred.image, addUserToPred.username)
   return response

@app.post("/createUser", response_model=model.user)
async def createUser(user: model.user):
   #if username is < 4 characters long
   if (len(user.username) < 4):
      raise HTTPException(status_code=409, detail="username requires 4 or more characters.")
   
   #if username is taken
   checkUsername = await database.fetch_user_by_username(user.username)
   if (checkUsername is not None):
      raise HTTPException(status_code=409, detail="username already exists.")

   #if password is < 6 long
   if (len(user.password) <= 6):
      raise HTTPException(status_code=409, detail="password is too short.")
   
   #does password contain numbers?
   passwordRegex = "^(?=.*[0-9].*[0-9])[a-zA-Z0-9]{6,}$"
   if not (re.fullmatch(passwordRegex, user.password)):
      raise HTTPException(status_code=409, detail="password must contain at least 6 characters and 2 numbers.")

   #is email valid?
   emailRegex = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
   if not (re.fullmatch(emailRegex, user.email)):
      raise HTTPException(status_code=409, detail="email not valid.")
   
   result = loadRunModel.createUser(user.username, user.email, user.password)
   database.create_user(result)
   return result

@app.post("/loginUserAuth", response_model=model.userLogin)
async def loginForAccessToken(form_data: OAuth2PasswordRequestForm = Depends()):
   user = await database.fetch_user_by_username(form_data.username)
   if not user:
      raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    
   encrypted_password = user.get("password")
   if not loadRunModel.comparePasswords(form_data.password, encrypted_password):
      raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    
   access_token_expires = datetime.timedelta(minutes=int(val.access_token_expire_time))
   access_token = loadRunModel.createAccessToken(data={"sub": form_data.username}, expires_delta=access_token_expires)
   return {"access_token": access_token, "token_type": "bearer"}

@app.get("/allUsers")
async def allUsers(): 
   response = await database.fetch_all_users()
   return response

@app.get("/getUser")
async def getUser(username):
   response = await database.fetch_user_by_username(username)
   if (response is None):
      raise HTTPException(status_code=409, detail='username does not exist')
   else:
      return response
   
@app.get("/userPredictions")
async def getUserPredictions(token: str):
   username = loadRunModel.verifyToken(token)
   response = await database.fetch_predictions_from_users(username)
   return response

@app.get("/userPredictionsNewest")
async def getUserPredictionsNewest(token: str):
   username = loadRunModel.verifyToken(token)
   response = await database.fetch_predictions_from_users_by_newest_date(username)
   return response

@app.get("/userPredictionsOldest")
async def getUserPredictionsOldest(token: str):
   username = loadRunModel.verifyToken(token)
   response = await database.fetch_predictions_from_users_by_oldest_date(username)
   return response

@app.delete("/deleteUserPrediction")
async def deleteUserPrediction(username: str, image: str):
   #username = loadRunModel.verifyToken(token)
   response = await database.delete_prediction(username, image)
   return response
