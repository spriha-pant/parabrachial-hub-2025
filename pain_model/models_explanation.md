Parameters:

- two-phase formalin pain curve:
    - pain line (blue) should have one major spike to around 0.4 pain value for the first 0-10 minutes, then drop near 0 pain, then rise again till 0.1 while slowly decaying.
    ideal look (paper):
    ![Alt text] (/home/arnablab/projects/Pain_Effort_RL/parabrachial-hub-2025/pain_model/ideal_output_images/ideal_pain.png_)
    - effort line (orange) should spike at 0-5 minutes (mouse is licking vigorously during the first phase of formalin pain), before stopping entirely, then spiking again after around 20 minutes to respond to the later formalin phase with licks of slightly less frequency.
    ideal look (paper):
    ![Alt text] (/home/arnablab/projects/Pain_Effort_RL/parabrachial-hub-2025/pain_model/ideal_output_images/ideal_effort.png)

- episode learning curve:
    There should be:
    - postive trend curve over iterations (no oscillation without positive learning)
    - high reward score, especially by the end of training
    close to ideal look:
    ![Alt text] (/home/arnablab/projects/Pain_Effort_RL/parabrachial-hub-2025/pain_model/ideal_output_images/ideal_learning.png)

- licking bar graph:
    - lines only at the beginning and in the middle, signifying that mouse licks ie. puts in effort according to accurate effort-pain graph.
    close to ideal look:
    ![Alt text] (/home/arnablab/projects/Pain_Effort_RL/parabrachial-hub-2025/pain_model/ideal_output_images/ideal_licking.png)

- green policy heatmap graph:
    - a combination of green effort patches versus distributed pain trajectory, rather than overwhelming amounts of green or excessive pain lines or blank heat map lacking in policy.
    close to ideal look:
    ![Alt text] (/home/arnablab/projects/Pain_Effort_RL/parabrachial-hub-2025/pain_model/ideal_output_images/ideal_policy.png)

Important formulae in train_P:
- Weightage to energy and pain (at yy and xx respectively)
- yy is how fast energy decays -> how fast mouse recovers from lick fatigue/effort ie. the fact that energy does not accumulate into something big and daunting to mouse; high yy makes it easier to lick often, low yy makes it less attractive to lick often.
- xx is... idk for sure, I think how much effort/licking contributes to reward?
So high xx means energy/effort matter more ie. if you invest in licking then the less pain felt and to not invest means feeling much worse.
    - conv_param = np.array([0.xx, 0.yy])
- Licking effect (at z):
    - x = x + (-x+pain)/5 + P_input - 0.00z*action + 0.001
    - x = x + (-x+pain)/5 + P_input - 0.00z + 0.001
- Energy accumulation (at xx):
    - energy = energy - energy*0.xx + action


What each iteration of models does right:
- two-phase formalin pain curve: model 3
- episode learning curve: model 1
- licking bar graph: model 4
- green policy heatmap graph: model 2