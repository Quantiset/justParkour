## Inspiration

All of us are avid Minecraft gamers who spent countless hours playing the game growing up. However, excessive screentime and sedentary lifestyles over physical activity can lead to increased risk of depression and anxiety [1] and obesity or poor fitness [4]. Among Generation Alpha, screentime is at its highest ever, with 30% of children spending more than 10 hours per week on devices, and these kids are strongly associated with the risks of poor fitness and mental health [2]. 

Simultaneously, exergames (active video games) show a strong correlation with energy expenditure and achieve moderate-intensity physical activity levels, meeting the American College of Sports Medicine guidelines for health and fitness [3]. Thus, we propose JustParkour, an interactive application that turns any Minecraft Parkour YouTube video into a fun Just Dance game to engage younger audiences in vigorous physical activity and make screen time more positive.

## What it does

JustParkour is an application that allows you to upload a Minecraft Parkour YouTube video and transform it into an interactive JustDance-like game. It enables 1 to 5 participants to run and jump along with the parkour video and scores them on a leaderboard based on their jump timing and running cadence. It also provides summary exercise feedback at the end of the session and vocal feedback for good exercise habits, tips, and positive reinforcement.

## How we built it
![System design diagram](https://drive.google.com/uc?export=view&id=1sRLG5c1K5livj-PWlaD5zkkJVPfn1bn2)
Our system design encapsulates the guiding design principles we emulated during our development process: sustainability and resource use optimization by caching expensive machine learning inference calls; support for frequent iterated design changes via API-first development; security for memory and API calls; and access to extensive resources via AWS.

Our frontend uses PyGame to seamlessly integrate the parkour video with the leaderboard and computer vision tracking. The leaderboard's scoring updates based on how accurately the participants jump in the parkour video, and checks whether they are actively running along with the game using MediaPipe. 

Our backend processes the YouTube video that the participant provides using an EC2 instance running Python with a state-of-the-art inverse dynamics model (IDM). This allows prediction of the actions likely taken given an input sequence of frames, which in our case predicts the actions the parkour runner will take from the previous Minecraft frames (specifically jumping).

## Reflection and What's Next

We learned how to leverage AWS EC2 instances, Gemini, and ElevenLabs to create an interactive experience. We overcame many challenges in classifying temporal data in videos and in integrating our product using AWS resources, and in the future, we plan to expand the project with different games and exercise strategies to promote fun and health for good.

## Citations

[1] B. Zablotsky, A. E. Ng, L. I. Black, G. Haile, J. Bose, J. R. Jones, et al., "Associations Between Screen Time Use and Health Outcomes Among US Teenagers," Preventing Chronic Disease, vol. 22, p. 240537, Jul. 2025, doi: 10.5888/pcd22.240537.

[2] W. A. Mohsen, M. Al-Rashaida, and A. M. Alkaabi, "Navigating Generation Alpha in the Digital Age: Parental Surveillance and Children's Online Engagement," Social Sciences & Humanities Open, vol. 12, art. no. 101875, Jan. 2025, doi: 10.1016/j.ssaho.2025.101875.

[3] J. Sween, S. F. Wallington, V. Sheppard, T. Taylor, A. A. Llanos, and L. L. Adams-Campbell, "The role of exergaming in improving physical activity: A review," Journal of Physical Activity and Health, vol. 11, no. 4, pp. 864–870, May 2014, doi: 10.1123/jpah 2011-0425.

[4] A. Sanyaolu, C. Okorie, X. Qi, J. Locke, and S. Rehman, "Childhood and adolescent obesity in the United States: A public health concern," Global Pediatric Health, vol. 6, pp. 1–11, Dec. 2019, doi: 10.1177/2333794X19891305.

## Images

![Gameplay](https://drive.google.com/uc?export=view&id=1_NCo6pIWcFN3Gg1369w3W-ksrCcM0eIm)
![Gameplay](https://drive.google.com/uc?export=view&id=17YGDFegJIM2Pqk2qoIx9pZ9So6uQwkBZ)
