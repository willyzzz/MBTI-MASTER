# MBTI-MASTER
First run the crawler code, then can collect dataset which is already in the data package.

The code in the preprocess package can help preprocess the data.

And there are codes for modelling in the modelling package


How to run chatbot:
The models need to be connected to the backend first.

Open the command prompt and follow the following commands:

cd "PchatbotVfin2"
git init
git add .
git commit -m "Heroku commit"
heroku login (Login account: sgzpt1215@gmail.com, Password: 1q2w3e4r5tAAA**)
heroku git:remote -a chatbot2p
git push heroku master

When finished, the chatbot is ready. 
