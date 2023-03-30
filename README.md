# DeepRL_Unity

Prosthetic vision for the blind: Testing trained computer agents for phosphene vision in a realistic environment 
  * training a Deep Reinforcement Learning agent to test the performance in particular tasks such as navigation and obstacle avoidance in a realistic virtual environment developed in Unity.

Thanks to Burcu Kucukoglu, Sam Danen, and Jaap de Ruyter van Steveninck for their contribution to the codes.


##### PyTorch
##### Unity

-------------

#### The aims of this research project with following provisional implications are: 
  1) Using a realistic environment to train and test the RL agent with phosphene vision. This will clear the way for studies with more complex environments and tasks such as dynamic environments. In line with this, using a realistic virtual environment will widen the choices of experimental designs. Additionally, the realistic environment developed in this study will provide a baseline testing environment for future studies. 
  2) Comparing different parameters to train the agent, which contributes to the aim of training equivalent Deep RL agents to human participants. Achieving this might decrease the costs significantly and facilitate studies in this area of research. 
  3) Comparing different image pre-processing techniques which might allow us to see the effect of extracting visual cues and the contribution of different visual cues in a realistic setting for a navigation task.
  
------------
  
 #### - Point Goal Navigation and Obstacle Avoidance
 :arrow_right: The agent was generated at the same location at the beginning of each episode and the task of the trained agent was to freely navigate in the environment towards the target by avoiding the wall and object collisions and by using the shortest path.
  
 #### - Realistic Virtual Environment 
 :arrow_right: Unity - ArchVizPro Interior Vol.1 3D Environment
  
  <img width="565" alt="env" src="https://user-images.githubusercontent.com/87897577/228913381-3aea00e7-4939-4073-8497-7d509765ab13.png">

 #### - Vision Processing
 
  * Phosphene Vision with Canny Edge Detection 
  
  <img width="502" alt="image" src="https://user-images.githubusercontent.com/87897577/228914612-3cfc1c7e-35b1-4fc7-8680-5dd4e18cc886.png">
  
  
  * Phosphene Vision with Object Contour Segmentation and Canny Edge Detection
  
  <img width="501" alt="image" src="https://user-images.githubusercontent.com/87897577/228914730-fc650148-77d5-44bb-a0df-a2afc61c0b50.png">

#### - Agent types depending on inputs 
   * Double-DQN Sighted Agents (Input: Gray-Scale Images)
   * Double-DQN Canny Agents (Input: Phosphene images after applying Canny Edge Detection)
   * Double-DQN Segmentation Agents (Input: Phosphene images after applying Object Contour Segmentation and Canny Edge Detection
   * Random Agent (Agent choose actions randomly)


- For dateails, please refer to my thesis project https://drive.google.com/file/d/1BpSDmnCU93v66h5bXSf1Cv3wfhMBIjIc/view
