import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pyplot

def huber_loss(labels, predictions, delta=14.0):
    residual = tf.abs(labels - predictions)
    def f1(): return 0.5 * tf.square(residual)
    def f2(): return delta * residual - 0.5 * tf.square(delta)
    return tf.cond(residual < delta, f1, f2)

def linear_regression(data, lossType):
    X = tf.placeholder(tf.float32, name="X")
    Y = tf.placeholder(tf.float32, name="Y")

    w = tf.get_variable(initializer=tf.constant(0.0), name="weights")
    b = tf.get_variable(initializer=tf.constant(0.0), name="bias")

    Y_predicted = w * X + b
    if(lossType == "huber"):
        loss = huber_loss(Y, Y_predicted)
    else:
        loss = tf.square(Y - Y_predicted)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    writer = tf.summary.FileWriter('./graphs/linear_reg', tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(100):
            total_loss = 0
            for x,y in data:
                _, l = sess.run([optimizer, loss], feed_dict={X: x, Y:y})
                total_loss += l
            print('Epoch {0}: {1}'.format(i, total_loss/n))
        
        writer.close()

        [w_out, b_out] = sess.run([w, b])
    
    tf.get_variable_scope().reuse_variables()

    return [w_out, b_out]


data = np.array([
[1, 12.0],
[2, 22],
[3, 27],
[4, 36],
[5, 53],
[6, 67],
[7, 66],
[8, 200],
])

n = 8

[w_out_squared, b_out_squared] = linear_regression(data, "squared")
[w_out_huber, b_out_huber] = linear_regression(data, "huber")

pyplot.plot(data[:,0], data[:,1], 'go', label='Real data')
pyplot.plot(data[:,0], data[:,0] * w_out_squared + b_out_squared, 'r', label='Predicted data with Squared Error loss')
pyplot.plot(data[:,0], data[:,0] * w_out_huber + b_out_huber, 'b', label='Predicted data with Huber loss')
pyplot.legend()
pyplot.show()

