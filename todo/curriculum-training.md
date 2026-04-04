I ran python train_alphazx.py --num-qubits 5 --depth 10 --num-iterations 100 --device cpu --log-level INFO and training 
appears to be progressing. I am concerned that, because the initial generated diagrams are non-trivial, the agent will 
never take enough chances randomly to simplify a diagram and get a reward. I think you mentioned something about 
curriculum training at an earlier point. The idea would be to start generating small diagrams and progressively make 
them larger through training. That way, the agent stands a chance of simplifying the simpler diagrams and progressively 
develops tactics for simplifying larger diagrams. Is this approach best-practice for my kind of problem? Would you 
recommend this approach? You don't have to agree with me.
