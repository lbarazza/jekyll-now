---
layout: post
title: An Introduction to Reinforcement Learning
---

Let's say that when you were a child you touched fire one time. After that, you realized it was probably a good a idea to not do it again and that's what you kept on doing ever after.
What is the mechanism that led your brain to understand that touching fire was not the best thing you could do? How could we develop a system that develops this same ability of interacting with an environment and learning what is good and what is not?

Reinforcement learning is the part of artificial intelligence that attempts to answer just that: what makes an intelligent being learn from interaction?
This is also known as the Reinforcement Learning Problem.

RL (Reinforcement learning) is part of machine learning, but it doesn't fall into either the category of supervised nor unsupervised learning. As of today, RL is the part of AI that is considered to come closer to achieving human-like intelligence.

### MDPs
A Markov Decision Process (MDP) is the formal framework used to address the Reinforcement Learning Problem. An MDP lays out a standardized way to formulate the problem, via an interaction between the agent (the intelligent being) and the environment in which the agent lives. In this interaction between agent and environment, the agent observes the state it is in, decides to take a certain action based on this state and then the environment responds with a new state and a reward, where the reward is a number defining how good that action was (a positive reward is good and a negative one is bad). This interaction can either have an end (in which case it is called an episodic task) or keep on going forever (in which case it is called a continuous task).
For example, in a video game you could think of the reward as being the score after you take the action. <As anther example we could say that the negative reward in a human being's life would be pain and the positive one would be pleasure>?.
In such a framework, it makes sense to assume that an agent's goal would be to maximize the future reward, that is the sum of all the rewards that will come next.

The Reward Hypothesis is the hypothesis that every goal or purpose can be described as the maximization of a single scalar, the future reward.
read more about it [here.](http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/rewardhypothesis.html)

What exactly is reward?
Above we defined the future reward as the sum of all the rewards that come next. This is called undiscounted future reward and in formula this would be:

   [formula for undiscounted future reward]

This implementation works well as long as we are dealing with an episodic task. In fact, if the task was not episodic then it would go on forever and the future reward could potentially diverge to infinity as well, which would prevent us from being able to maximize it.
In the cases where we are dealing with a continuous task, we then have to come up with something else. The solution is to just slightly change the above formula so that the more a reward is in the future, the more it gets "discounted", so that it will eventually converge to a finite value (this variation of the formula is called discounted future reward). We can achieve this by adding a constant gamma γ ∈ [0, 1] which we raise to the nth power where n represents the number of time steps in the future.

   [formula for undiscounted future reward]

In the case in which γ = 1, we get back the undiscounted formula.


### Policies and Value Functions
Now that the general framework has been defined, we can start thinking about everything in more detail.
For example, we said that the agent takes an action based on the state that it observes, but how exactly does it do that? The answer is that it follows a policy. A policy is a function that maps a certain state to the probability of taking each action.

   [policy definition formula]

But how do we find this policy? It turns out that it is useful to define two new functions first, the state-value function and the action-value function. The state-value function tells you the expected return that the agent would get if it started at the state s and followed the policy π for all future time steps. Formally,

https://wikimedia.org/api/rest_v1/media/math/render/svg/cdc6b4354d612c08f5009fe29432d6779036343b

This function also has a particularly useful property which is summarized by the Bellman Expectation Equation

  [bellman expectation equation]

Where, the only thing different from the last equation is that R (the future reward) is expressed in its expanded form. If we then expand the expectation we get,

  [bellman expectation equation expanded form]

Now that we have the state-value function, we can define the action-value function. This latter function (often denoted with q) is very similar to the state-value function, but, instead of telling you the expected return of following a certain policy starting from a certain state, it tells you the expected return of starting at a certain state s and taking an action a (which doesn't have to be the one the policy would choose) and then following the policy ever after. The advantage of defining this function is that if we manage to "find" it, we can then improve our current policy by constantly choosing the action that maximizes the q-function.

### Model-Based Methods
##### Dynamic Programming
So far we only described some theoretical pre-requisites, but we never really described how to solve the problem we set for ourselves, that is find a policy that maximizes the expected future reward. Here I describe how to do just this.
We will start off with an assumption that simplifies the problem and then we will discuss other methods that don't require us to make this assumption. The assumption is that we have complete knowledge of the MDP, that is we know with what probability an action will yield a certain state and we also know the reward that we would get from such transition (algorithms that work with this assumption are called model-based methods). 
If we have this knowledge we can then find the optimal policy with a method called Dynamic Programming.
This approach bases itself off the realization that, by solving the system of equations made up by the bellman equations for the value function at every state in the MDP, we end up with the optimal state-value function. When we have the optimal state-value function we can then easily derive the q-function (action-value function) with this formula,

As showed earlier, having the q-function allows us to easily find the best policy for the state-value function.

If we have complete knowledge of he MDP, we can always solve the system of linear equations, but this is usually practically inefficient as the number of states is often very large. To avoid this we can take a slightly different approach called iterative policy evaluation. This consists of initializing each state with a certain value and then looping through each state and assigning to each a new value that you get from the Bellman equation,

 [bellman update equation for iterative policy evalutation]

This way, our approximation for the function will keep on changing until it reaches an equilibrium point where it doesn't change that much anymore. At that point, it has sufficiently approximated the state-value function so we exit the loop. Of course this method will hardly ever give us an exact representation of v, but we can get as good of an approximation as we want by stopping the approximation whenever the change in v is smaller than a hyperparameter we set θ (theta).
This method only tells us the v function corresponding to the current policy, when what we really want is the v function for the optimal policy. But now that we have the v function we can derive the q-function as described above and then get a new improved policy from there. At this point we evaluate the v function for the new policy and then, again, derive the q-function and get an even better policy and so on. This whole process is called policy iteration and is composed of policy evaluation (finding the current v-function) and policy improvement (getting the new improved policy). We will come across policy iteration many times in RL.

### Model-Free Methods

##### Monte Carlo Methods
In the above section we made the assumption that we know everything about how our model behaves, but this is often not the case so we need to come up with something else for all the times where this assumption is not feasible.
One way to do this is to use a genre of methods called Monte Carlo Methods. Before our objective was to find the state-value function as we knew that if we managed to find that, we could then derive the q-function and then, finally, the policy. In this case, though, we are generally not able to get the q-function from the v-function as we don't know the details about the MDP (such as the transition rewards and probabilities) which we needed in the formula we used before. One way to get around this is to directly learn the q-function, without passing through the state-values.
To do this we just run our agent in the environment and register every state and reward. At the end of the nth episode we then calculate the future reward (discounted or undiscounted) for every state-action pair visited (that we recorded) and average it over the number of times we visited that state. This will give us an estimate of the true action-value for that state.
Now we run into two different possibilities, either to consider every single visit to a state-action pair in one episode or to just consider the first one. In the first case we have the so called Firs-Visit MC Method and in the latter we have the Every-Visit MC Method.

This approach works and doesn't need knowledge of the model, but it still has many limitations. One of which is that it has to run through a number of episodes before being able to update the Q-table, which is very inefficient. One way to do this is to use a running average (Incremental Mean Method),

  [running average formula] 

but this means that the latest episodes will have less and less weight as the number of total episodes increases. This isn't the best, though, because the latest episodes are the ones that have been performed with a better policy and are the ones that should have more weight. To overcome this we can use the constant-alpha variation of the above formula,

[constant-alpha formula]

where, instead of simply averaging over all values, we take an exponential moving average, to give more weight to the latest episodes. α (alpha) ∈ (0, 1] is the learning rate of the function, the bigger α the faster the agent will learn, but if α is too big, then the agent we would keep overshooting our approximation for the q values and the agent would never learn.

##### GLIE
These two methods as they have been described above won't work. To understand why, I need to introduce the problem of exploration vs. exploitation. This simply says that if we always choose the action that maximizes the q-function we might get stuck in a sub-optimal policy. This is because the agent would never be willing to try new actions that on paper have worse value than the current "optimal" (exploitation), but that in reality are better and are only badly represented in the Q-table because they have been tested out poorly. On the other hand, if we allow our agent to choose randomly too much to avoid the problem of being stuck in a sub-optimal policy (exploration), it would also never be able to perform well as, even though it knows what action would be best, it still will often perform another because of it having to choose random actions to explore.
Finding a good balance between exploration and exploitation is still a hot topic of study, but for simplicity, it can be summarized with GLIE. GLIE stands for Greedy in the Limit with Infinite Exploration.
If ε (epsilon) is the exploration rate (the probability that the agent takes a random action to explore), then GLIE says that ε should decrease with time, eventually converging to zero as the number of episodes goes towards infinity. This way the agent will always explore, but, in the limit, it will converge towards a greedy policy (as ε -> 0).

 [two formulas for GLIE]

##### Temporal Difference Methods
The MC methods we discussed above have a few problems: they don't work for continuous tasks as MC methods need to wait for the end of the episode to update the Q-table (which also makes them very inefficient).
That's were Temporal Difference (or TD) methods come in. TD methods are able to update the Q-table at every time step in an episode. To do this they, again, make use of the Bellman equation, making use of their current predictions to approximate the expected future reward of a state.

There is a variety of TD methods such as Sarsa, Sarsamax (or q-learning), and Expected Sarsa. The simplest, Sarsa, takes an action, observes the state and the reward, then it chooses a new action based on the new state. It then uses the current estimate in the Q-table for the new action-value pair and observed reward to update the Q-table for the original action-value pair.
 
 [bellman equation for sarsa]

Expected Sarsa is an improvement of Sarsa which, instead of using the value of the action the policy would choose after observing the results of the first action, it makes a weighted average of all the action-values from that observation (the expected future reward of taking action a at state s).

 [bellman equation for expected sarsa]

Q-learning (or Sarsamax) is another variation of Sarsa where the only difference is that we don't consider the action-value of the next action the policy would choose, but, instead, the action that maximizes the future reward of the policy at that state. In other words, we are choosing our action based on a greedy policy.

 [bellman equation for q-learning]

This doesn't look like much, but it actually has important consequences. In fact, this moves the policy improvement off-policy, which means that it directly improves the policy with the optimal policy in mind.

You can find an implementation of three TD methods on my github page [here](https://github.com/lbarazza/Taxi-v2).

### How to Deal with Continuous State Spaces
So far all we did to represent the q-function in our code was to use a table of data. This, though, seems to be a problem when we start to explore continuous state-spaces as we would need an infinite amount of entries in our Q-table to represent the q-function completely.
There are generally two methods to deal with this problem: state-space discretization and function approximation.

##### State-Space Discretization
The most trivial way to deal with the problem is to discretize the state-space. This can be achieved by creating a grid on the state-space and let each point be represented by the tile of the grid it falls into. There is also a more advanced way of discretizing the state-space which makes use of multiple grids overlapping one another. You can find an implementation of this approach called Tile Coding on my Github [here](https://github.com/lbarazza/Tile-Coding).

##### Function Approximation
Another approach to the problem of continuous state spaces is to try to find a function which directly approximates the q-function. We can do this by defining a parametrized function q(s, a, W) which adjusts the parameter W to  try to find an approximation of q. As you may recognize this is nothing more than a neural network and it's right here where deep reinforcement learning comes in.

### Conclusion
After reading this post, you should have a general idea of the main concepts in reinforcement learning which should allow you to better understand the foundations of the very exciting field of deep reinforcement learning which is changing the world with algorithms such as AlphaGo Zero.

