## Training

loading data: The processed data should be stored in .mm and .txt. See [the tutorial of data preprocesing](https://github.com/yiruiliu110/ConditionalVI/blob/master/tutorial/data_processing.md) for details.

Please note that this dataset that contains 1740 documents is only used for tutorial. The real application of HDP required much larger dataset.


```python
from gensim import corpora
corpus = corpora.MmCorpus("tutorial/nips.mm")
```


```python
id2word = corpora.Dictionary.load_from_text("tutorial/nips_wordids.txt")
```


```python
print(corpus)
```

    MmCorpus(1740 documents, 8013 features, 613693 non-zero entries)
    

train and test split


```python
from sklearn.model_selection import train_test_split
```


```python
index_train, index_test = train_test_split(list(range(len(corpus))), test_size=200)
```

We are ready to train the HDP model. 


```python
from src.CATVI import HdpModel_CATVI
```


```python
model = HdpModel_CATVI(corpus=corpus[index_train], corpus_test=corpus[index_test], id2word=id2word)
```

fit the model


```python
model.fit()
```

       iter_no     K  time  likelihood  perplexity
    0      0.0  26.0  58.0   -8.635102     5625.71
        iter_no     K   time  likelihood  perplexity
    50     50.0  33.0  918.0   -8.402752     4459.32
         iter_no     K    time  likelihood  perplexity
    100    100.0  41.0  1851.0     -8.3794     4356.39
         iter_no     K    time  likelihood  perplexity
    150    150.0  51.0  2808.0   -8.369146     4311.95
         iter_no     K    time  likelihood  perplexity
    200    200.0  61.0  3770.0   -8.363219     4286.47
         iter_no     K    time  likelihood  perplexity
    250    250.0  67.0  4728.0   -8.358832     4267.71
         iter_no     K    time  likelihood  perplexity
    300    300.0  72.0  5671.0   -8.355677     4254.26
         iter_no     K    time  likelihood  perplexity
    350    350.0  77.0  6631.0   -8.352759     4241.87
         iter_no     K    time  likelihood  perplexity
    400    400.0  80.0  7592.0   -8.350632     4232.86
    

print the topic_word clustering result


```python
model.output()
```




    [(0,
      '0.003*stimulus + 0.003*circuit + 0.003*mixture + 0.002*motion + 0.002*synaptic + 0.002*spike + 0.002*classifier + 0.002*filter + 0.002*analog + 0.002*chip + 0.002*recurrent + 0.002*firing + 0.002*eye + 0.002*pixel + 0.002*cortex + 0.002*orientation + 0.002*expert + 0.002*cluster + 0.002*delay + 0.002*voltage'),
     (1,
      '0.006*theorem + 0.006*tree + 0.004*regression + 0.004*classifier + 0.004*kernel + 0.004*bayesian + 0.004*loss + 0.003*proof + 0.003*polynomial + 0.003*hypothesis + 0.003*posterior + 0.002*generalization_error + 0.002*validation + 0.002*wi + 0.002*entropy + 0.002*perceptron + 0.002*margin + 0.002*risk + 0.002*concept + 0.002*vc'),
     (2,
      '0.013*policy + 0.011*reinforcement + 0.008*robot + 0.007*reinforcement_learning + 0.007*controller + 0.006*reward + 0.005*td + 0.005*markov + 0.005*sutton + 0.004*trajectory + 0.004*agent + 0.004*trial + 0.004*transition + 0.004*path + 0.004*programming + 0.003*grid + 0.003*barto + 0.003*strategy + 0.003*game + 0.003*state_space'),
     (3,
      '0.007*circuit + 0.006*query + 0.006*template + 0.006*student + 0.005*conditional + 0.004*impulse + 0.004*grammar + 0.003*string + 0.003*window + 0.003*recurrent + 0.003*em + 0.003*inference + 0.002*pointing + 0.002*c1 + 0.002*entropy + 0.002*cm + 0.002*teacher + 0.002*robot + 0.002*chen + 0.002*policy'),
     (4,
      '0.008*crossing + 0.007*chip + 0.007*zero_crossing + 0.006*edge + 0.004*filter + 0.004*control_law + 0.004*voltage + 0.004*resistive + 0.003*sonar + 0.003*collection + 0.003*receptor + 0.003*analog + 0.003*vlsi + 0.003*backward + 0.003*intensity + 0.002*jr + 0.002*protein + 0.002*turning + 0.002*fragment + 0.002*edelman'),
     (5,
      '0.010*motion + 0.009*synaptic + 0.008*eye + 0.006*weak + 0.006*associative + 0.004*camera + 0.004*long_term + 0.004*hippocampus + 0.004*command + 0.003*synaptic_strength + 0.003*motor + 0.003*receptive_field + 0.003*visual_motion + 0.003*star + 0.003*rotation + 0.003*postsynaptic + 0.003*depression + 0.003*plasticity + 0.003*transistor + 0.002*burst'),
     (6,
      '0.024*perturbation + 0.005*alspector + 0.004*vlsi + 0.004*gradient_descent + 0.004*ll + 0.003*string + 0.003*learning_rate + 0.003*surface + 0.003*second_order + 0.003*asymptotic + 0.003*analog + 0.003*regional + 0.003*ae + 0.002*flower + 0.002*transforms + 0.002*decoder + 0.002*analog_vlsi + 0.002*wahba + 0.002*might_be + 0.002*morgan_kaufmann'),
     (7,
      '0.007*font + 0.005*circuit + 0.004*dictionary + 0.004*width + 0.003*preference + 0.003*alpha + 0.003*sorting + 0.003*rhythm + 0.003*boltzmann + 0.003*lower_bound + 0.003*voltage + 0.002*law + 0.002*unrestricted + 0.002*miller + 0.002*window + 0.002*recursion + 0.002*inconsistency + 0.002*partition + 0.002*tanh + 0.002*magnetic'),
     (8,
      '0.014*tree + 0.007*margin + 0.006*decision_tree + 0.005*flight + 0.005*utility + 0.004*structural + 0.004*fat + 0.003*wi + 0.003*parietal + 0.003*perceptton + 0.003*synaptic_transmission + 0.003*episode + 0.003*conditional_density + 0.002*ocular_dominance + 0.002*stereo + 0.002*relative_importance + 0.002*classified + 0.002*metric + 0.002*arrangement + 0.002*billion'),
     (9,
      '0.011*language + 0.008*decoder + 0.003*encoder + 0.003*evolved + 0.003*agent + 0.003*precision + 0.003*queue + 0.003*concept + 0.003*site + 0.002*symbol + 0.002*signal_processing + 0.002*inversion + 0.002*modulation + 0.002*ann + 0.002*krogh + 0.002*sonar + 0.002*material + 0.002*evolution + 0.002*planck + 0.002*learner'),
     (10,
      '0.007*weight_decay + 0.006*decay + 0.004*moody + 0.003*test_set + 0.003*regularization + 0.003*risk + 0.003*competitive + 0.003*anns + 0.002*regression + 0.002*squared + 0.002*barron + 0.002*sustained + 0.002*helmholtz_machine + 0.002*simulate + 0.002*tap + 0.002*drive + 0.002*randomized + 0.002*yi + 0.002*heart + 0.002*objective_function'),
     (11,
      '0.010*pyramid + 0.008*language + 0.008*compression + 0.006*writer + 0.005*food + 0.005*floating_gate + 0.004*markov_process + 0.003*interpolation + 0.003*conflicting + 0.003*mistake + 0.003*decoder + 0.002*concentration + 0.002*mixture + 0.002*infant + 0.002*friedman + 0.002*activation_function + 0.002*label + 0.002*manipulated + 0.002*weakness + 0.002*image_compression'),
     (12,
      '0.005*m + 0.005*gene + 0.003*pas + 0.003*linear_combination + 0.003*member + 0.003*ica + 0.002*limiting + 0.002*evaluation + 0.002*em + 0.002*sensory + 0.002*wavelet + 0.002*lazzaro + 0.002*phys + 0.002*receptive_field + 0.002*xu + 0.002*wk + 0.002*cascaded + 0.002*behavioral + 0.002*expert + 0.002*drift'),
     (13,
      '0.012*signature + 0.006*monotonicity + 0.005*sec + 0.003*altered + 0.003*vehicle + 0.003*implication + 0.003*hidden_layer + 0.003*successor + 0.002*graded + 0.002*trigger + 0.002*decorrelation + 0.002*analogue + 0.002*destination + 0.002*ring + 0.002*harmonic + 0.002*russell + 0.002*pc + 0.002*df + 0.002*female + 0.002*svm'),
     (14,
      '0.006*guidance + 0.005*compensation + 0.004*equilibrium + 0.003*resolution + 0.003*price + 0.003*phonetic + 0.003*boosting + 0.003*phonemic + 0.002*physical_review + 0.002*hyperparameters + 0.002*option + 0.002*simultaneous + 0.002*early_stopping + 0.002*adapt + 0.002*cascade + 0.002*willshaw + 0.002*suppressed + 0.002*gaussian_mixture + 0.002*monotonic + 0.002*outgoing'),
     (15,
      '0.007*expert + 0.005*event + 0.005*speaker + 0.004*bond + 0.004*competitive + 0.003*ng + 0.003*cohen + 0.003*verification + 0.003*cerebellar + 0.003*unsupervised_learning + 0.003*surround + 0.003*mixture + 0.003*fk + 0.003*jo + 0.003*ccd + 0.003*principal_component + 0.002*scattered + 0.002*equilibrium_point + 0.002*phase_locked + 0.002*don'),
     (16,
      '0.005*land + 0.005*classifier + 0.005*robot + 0.005*dna + 0.004*constancy + 0.004*recurrent + 0.004*vq + 0.004*convolution + 0.003*infomax + 0.003*oscillation + 0.003*trajectory + 0.003*soft + 0.003*transmission + 0.003*suppression + 0.002*reflex + 0.002*circular + 0.002*spring + 0.002*recurrent_network + 0.002*afferent + 0.002*ki'),
     (17,
      '0.005*graph + 0.005*cortical + 0.005*movement + 0.004*prototype + 0.003*second_order + 0.003*sensory + 0.003*trajectory + 0.003*food + 0.003*pc + 0.002*tangent + 0.002*cap + 0.002*oe + 0.002*inconsistency + 0.002*programmable + 0.002*generative_model + 0.002*hundred + 0.002*primary + 0.002*simard + 0.002*ghahramani + 0.002*adaptable'),
     (18,
      '0.011*star + 0.005*bifurcation + 0.004*contour + 0.003*efficiency + 0.003*cun + 0.003*xy + 0.002*ej + 0.002*ascent + 0.002*contralateral + 0.002*coincidence + 0.002*conflict + 0.002*policy + 0.002*macroscopic + 0.002*station + 0.002*positioning + 0.002*laplacian + 0.002*connection_strength + 0.002*orientation_tuning + 0.002*pas + 0.002*profile'),
     (19,
      '0.008*cone + 0.006*interspike + 0.003*grouping + 0.003*vertex + 0.003*list + 0.003*large_vocabulary + 0.003*epsp + 0.002*merging + 0.002*inferred + 0.002*mlp + 0.002*survival + 0.002*relevance + 0.002*motion + 0.002*recording + 0.002*hz + 0.002*cluster + 0.002*mead + 0.002*delta + 0.002*reaching + 0.002*flat')]




```python

```
