import numpy as np
import matplotlib.pyplot as plt

def std(P):
    x=P
    for i in range(100):
        x=x@P
    return x[0]

def generate_data(num_arms):
    # num_arms: number of arms
    # num_iter: number of iterations
    # return: regret
    num_states =2
    transition_prob=np.zeros((num_arms,num_states,num_states),dtype=np.float32)
    for i in range(num_arms):
        transition_prob[i][0][0]=np.random.uniform(0,1)
        transition_prob[i][0][1]=1-transition_prob[i][0][0]
        transition_prob[i][1][0]=np.random.uniform(0,1)
        transition_prob[i][1][1]=1-transition_prob[i][1][0]

    pi=np.zeros((num_arms,2)) #stationary distribution

    for i in range(num_arms):
        pi[i]=std(transition_prob[i])
    
    mean_reward=np.zeros((num_arms,2))

    # for i in range(int(num_arms//2)):
    #     mean_reward[i] = np.random.uniform(0,0.1,2)
    
    # for i in range(num_arms//2,num_arms):
    #     mean_reward[i] = np.random.uniform(0.9,1,2)
    
    # for i in range(num_arms):
    #     mean_reward[i] = np.random.uniform(0.4,0.6,2)
    
    for i in range(num_arms):
        mean_reward[i] = np.random.random(2)
    
    # for i in range(num_arms):
    #     mean_reward[i] = np.random.uniform(i/10,(i+1)/10,2)
    
    arm_mean_reward=np.zeros(num_arms) #mean reward of each arm

    for i in range(num_arms):
        arm_mean_reward[i]=pi[i]@(mean_reward[i].T)
    
    best_mean_reward=np.max(arm_mean_reward)

    return best_mean_reward,arm_mean_reward,mean_reward,pi,transition_prob

def mab_markov_ucb(num_arms,num_iter,L,best_mean_reward,arm_mean_reward,mean_reward,pi,transition_prob):
    # num_arms: number of arms
    # num_iter: number of iterations
    # L: a constant
    
    states = np.zeros(num_arms,dtype=np.int32)
    T = np.zeros(num_arms,dtype=np.int32)   #T[i] is the number of times arm i is pulled
    sum_rewards = np.zeros(num_arms)   #sum_rewards[i] is the sum of rewards of arm i
    regret = np.zeros(num_iter)
    g = np.zeros(num_arms) #g[i] is the index for arm i
    
    #We start with an exploring start playing all arms initially in the first iteration

    for arm in range(num_arms):
        reward = np.random.binomial(1,mean_reward[arm][states[arm]])
        sum_rewards[arm] += reward
        T[arm] += 1
        g[arm] = sum_rewards[arm]/T[arm]
        # print(transition_prob[arm][states[arm]][0][1])
        states[arm] = np.random.binomial(1,transition_prob[arm][states[arm]][1])
        # regret[0] += best_mean_reward - arm_mean_reward[arm]

    for iter in range(1,num_iter):
        selected_arm = np.argmax(g)
        reward = np.random.binomial(1,mean_reward[selected_arm][states[selected_arm]])
        sum_rewards[selected_arm] += reward
        T[selected_arm] += 1
        g = (sum_rewards/T)+np.sqrt(L*np.log(iter+1)/(T+1))
        states[selected_arm] = np.random.binomial(1,transition_prob[selected_arm][states[selected_arm]][1])
        regret[iter] = regret[iter-1] + best_mean_reward - arm_mean_reward[selected_arm]
    
    return regret

def kl_divergence(p,q): # KullBack Leiber distance
    
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def kl_bound(t,T,arm):  # The upper bound in the KL_UCB  algorithm
    return (np.log(1 + t*(np.log(t)**2)))/T[arm]


def mab_markov_kl_ucb(num_arms,num_iter,best_mean_reward,arm_mean_reward,mean_reward,pi,transition_prob):
    # num_arms: number of arms
    # num_iter: number of iterations
    # L: a constant
    
    states = np.zeros(num_arms,dtype=np.int32)
    T = np.zeros(num_arms,dtype=np.int32)   #T[i] is the number of times arm i is pulled
    sum_rewards = np.zeros(num_arms)   #sum_rewards[i] is the sum of rewards of arm i
    regret = np.zeros(num_iter)
    g = np.zeros(num_arms) #g[i] is the index for arm i
    
    #We start with an exploring start playing all  arms initially in the first iteration
    for arm in range(num_arms):
        reward = np.random.binomial(1,mean_reward[arm][states[arm]])
        sum_rewards[arm] += reward
        T[arm] += 1
        g[arm] = sum_rewards[arm]/T[arm]
        states[arm] = np.random.binomial(1,transition_prob[arm][states[arm]][1])
        # regret[0] += best_mean_reward - arm_mean_reward[arm]


    for t in range(1,num_iter):
        for arm in range(num_arms):
            exp_r=1 # initializing the value in the range of mean estimate and 1 that would satisfy the inequality 
            while exp_r>=(sum_rewards[arm]/T[arm]): #run loop till we satisfy the inequality
                print("t=",t,"arm=",arm,"exp_r=", exp_r)
                v1 = kl_divergence(sum_rewards[arm]/T[arm],exp_r)
                v2 = kl_bound(t,T,arm)
                if v1<=v2:
                    g[arm]=exp_r
                    break
                exp_r-=0.1
      
        selected_arm = np.argmax(g)
        reward=np.random.binomial(1,mean_reward[selected_arm][states[selected_arm]])
        T[selected_arm]+=1
        sum_rewards[selected_arm]+=reward
        states[selected_arm] = np.random.binomial(1,transition_prob[selected_arm][states[selected_arm]][1]) #transition to state 0 with probability trans_prob or else be in state 1

    # if t==0:
    #   regret[t]=best_mean_reward-arm_mean_reward[selected_arm]
    # else:
        regret[t]=regret[t-1]+best_mean_reward-arm_mean_reward[selected_arm] #calculating regret
    
    return regret


def plot_regret(regret_ucb, regret_kl_ucb):

    figure, axis = plt.subplots(2)
  
    axis[0].plot(regret_ucb)
    axis[0].set_title("UCB Regret")
    
    axis[1].plot(regret_kl_ucb)
    axis[1].set_title("KL-UCB Regret")
    
    plt.show()
    

#call the functions
num_arms=10
num_iter=10000
L=2
best_mean_reward,arm_mean_reward,mean_reward,pi,transition_prob=generate_data(num_arms)
print(arm_mean_reward)

regret_ucb=mab_markov_ucb(num_arms,num_iter,L,best_mean_reward,arm_mean_reward,mean_reward,pi,transition_prob)

regret_kl_ucb=mab_markov_kl_ucb(num_arms,num_iter,best_mean_reward,arm_mean_reward,mean_reward,pi,transition_prob)

plot_regret(regret_ucb, regret_kl_ucb)