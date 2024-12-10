import numpy as np
from numpy.linalg import inv
import time
def filter(params, UpdateFlg, dt, model, Xkminus1, Pkminus1, Uk, sensors, Yks):
    params = params.astype(np.float64)

    # Update...
    (Xk, Pk) = update(params, UpdateFlg, Xkminus1, Pkminus1, Uk, sensors, Yks)
    # Predict...
    (Xk, Pk) = predict(params, dt, model, Xk, Pk, Uk)


    # Predict 1st and Update 2nd???...
    # (Xk, Pk) = predict(params, dt, model, Xkminus1, Pkminus1, Uk)
    # (Xk, Pk) = update(params,UpdateFlg, Xk, Pk, Uk, sensors, Yks)

    return (Xk, Pk)
    # return (Xkminus1, Pkminus1)

def predict(params, dt, model, Xkminus1, Pkminus1, Ukminus1):
    # start_t = time.time()
    (f, Q) = model
    # try:
    (fx, Fx) = f(params, dt, Xkminus1, Ukminus1)
    # except:
    #     print('ERR In PREDICT!')
    fx = fx[0:, 0]
    Fx = Fx.astype(np.float64)
    # print('Predicted State Estimate: \n %f' %(time.time() - start_t))

    # start_t = time.time()
    Xk = fx
    Pk = np.dot(Fx, np.dot(Pkminus1, Fx.T)) + np.dot(dt**2, Q)

    # Ensure Covariance matrix is symmetrical
    # https://stackoverflow.com/a/30010778/4255176
    Pk = (Pk + np.transpose(Pk)) / 2
    # print('Predicted Estimate Covariance: \n %f' %(time.time() - start_t))
    return (Xk, Pk)

def update(params, UpdateFlg, Xk, Pk, Uk, sensors, Yks):

    for (sensor, h, R) in sensors:
        if sensor in Yks.keys():
            Yk = Yks[sensor]

            if UpdateFlg[sensor]:
                # start_t = time.time()
                (hx, Hx) = h(params, Xk, Uk)
                hx = hx[0:, 0]
                Hx = Hx.astype(np.float64)
                # print('Linearization Time: \n %f' %(time.time() - start_t))

                # start_t = time.time()
                S = Hx.dot(Pk).dot(Hx.T) + R
                # print('Inovation Time: \n %f' %(time.time() - start_t))

                # start_t = time.time()
                invS = np.diag(1./np.diag(S))  # Approximation of S by only keeping diagnal terms
                K = Pk.dot(Hx.T).dot(invS)
                # K = Pk.dot(Hx.T).dot(inv(S))


                # try:
                #     K = np.linalg.solve(S.T, Pk.dot(Hx.T).T).T
                # except:
                #     print('ERROR FOUND!!!!')
                # print('Optimal Kalman gain Time: \n %f' %(time.time() - start_t))


                # start_t = time.time()

                v = K.dot(Yk - hx)
                # GPS YAW STUFF
                if (sensor == "gps"):
                    phasediff = v[3]
                    if (np.abs(phasediff) > np.pi):
                        v[3] = np.mod(phasediff, -np.sign(phasediff) * 2*np.pi)

                Xk = Xk + v

                # print('Updated state estimate: \n %f' %(time.time() - start_t))

                # start_t = time.time()

                Pk = Pk - K.dot(S).dot(K.T)


                # Ensure Covariance matrix is symmetrical
                # https://stackoverflow.com/a/30010778/4255176≈
                Pk = (Pk + np.transpose(Pk)) / 2
                # print('Update Estimate Covariance: \n %f' %(time.time() - start_t))
    return (Xk, Pk)
