---
layout: page
title: "Tensorflow-Light Field From Single Image"
---

- [Exponential Moving Average (EMA)](#ema)  
  
## <a name="ema"></a> Exponential Moving Average (EMA)
When training a model, it is often beneficial to maintain moving averages of the trained parameters. Evaluations that use averaged parameters sometimes produce significantly better results than the final trained values.   
> shadow_variable = decay * shadow_variable + (1 - decay) * variable  
```python  
ema = tf.train.ExponentialMovingAverage(decay=0.99)  
update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])  
```

