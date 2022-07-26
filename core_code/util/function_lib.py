from typing import Callable
import numpy as np

def function_library(
    function: str = 'sine', 
    p: list = [.4, 1., 2.0, -3.0]
) -> Callable:
    """Function library.

    Args:
        function (str, optional): String designation of the wanted function (refer to source code for options). Defaults to 'sine'.
        p (list, optional): Optional parameters supplied as list. Defaults to [.4, 1., 2.0, -3.0].

    Returns:
        _type_: _description_
    """

###########################################################################################################################
# 1d
###########################################################################################################################
#--------------------------------------------------------------------------------------------------------------------------
    if function == 'sine':
        return lambda x : np.sin(np.pi * x[:,0])
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'exp':
        return lambda x : np.exp(x[:,0]) * .7357 - 1
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'negToPos':
        return lambda x : np.squeeze(2.0 * (x[:, 0] > 0) - 1)
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'posToNeg':
        return lambda x : np.squeeze(2.0 * (x[:, 0] < 0) - 1)
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'squared':
        return lambda x : x[:,0]**2 - 0.5
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'scaledSquared':
        return lambda x : 2*(x[:,0]**2 - 0.5)
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'cubed':
        return lambda x : x[:,0]**3
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'oneKink':
        return lambda x : p[2] * (x[:,0] < p[0]) * (x[:,0] - p[0]) + p[1] + p[3] * (x[:,0] > p[0]) * (x[:,0] - p[0])
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'scaledoneKink':
        extremePoints=function_library(function = 'oneKink', p = p)(np.array([-1,p[0],1]).reshape(-1,1))
        ma=np.max(extremePoints)
        mi=np.min(extremePoints)
        return lambda x : (function_library(function = 'oneKink', p = p)(x)-(mi+ma)/2)*2/(ma-mi)
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'abs':
        return lambda x : np.abs(x[:,0] - 0.4) + .5
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'scaledAbs':
        return lambda x : np.abs(x[:,0] - 0.4)*2/1.4-1
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'brutalSine':
        return lambda x : np.sin(4.0 * 5 * x[:,0])
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'scaledbrutalSineTrend':
        return lambda x : np.sin(-np.pi * 5.5 * x[:,0])/5+x[:,0]*4/5
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'scaledForrester':
        return lambda x : (((6*((x[:,0]+1)*.5)-2)**2*np.sin(12*((x[:,0]+1)*.5)-4))+6.02074)/(16*np.sin(8)+6.02074)*2-1
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'scaledSine':
        return lambda x : np.sin(np.pi* p[-1] * x[:,0])
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'jakobhfun':
        return lambda x : np.sin(np.exp(2.7*x[:,0])+0.1)*((x[:,0]+1)/2)**(0.9)
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'scaledJakobhfun':
        return lambda x : (np.sin(np.exp(2.7*x[:,0])+0.1)*((x[:,0]+1)/2)**(0.9)+0.948031)/(0.990351+0.948031)*2-1
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'scaledJakobhfunTrend':
        return lambda x : (0.5*np.sin(np.exp(2.7*x[:,0])+0.1)*((x[:,0]+1)/2)**(0.9)+ 0.5*x[:,0] +0.25*(x[:,0]+1)**2-0.7324646)/(2.46492193)*2
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'brutalSine2':
        return lambda x : np.sin(10/(x[:,0]-1.2))
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'scaledbrutalSine2Trend':
        return lambda x : np.sin(10/(x[:,0]-1.21952))*0.25-0.25*x[:,0] -0.5*x[:,0]**3 #-0.25/4*(x[:,0]+1)**2
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'firstExperiment':
        return lambda x : np.array([function_library('oneKink')(x), function_library('abs')(x), function_library('sine')(x)]).T
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'scaledLevy':
        def Levy(x):
            w = 1 + (x - 1)/4
            return (np.sin(np.pi*w))**2 +(w-1)**2 * (1+1*(np.sin(2*np.pi*w))**2)
        scale=10
        ma=Levy(-10)
        mi=0.0
        return lambda x : (Levy(x[:,0]*scale)-(ma+mi)*0.5)*2/(ma-mi)
#--------------------------------------------------------------------------------------------------------------------------
    elif function == 'compositeSine':
        return lambda x : np.array([function_library('oneKink')(function_library('scaledSine',p=p)(x).reshape((-1,1))),
                                    function_library('abs')(function_library('scaledSine',p=p)(x).reshape((-1,1))),
                                    function_library('squared')(function_library('scaledSine',p=p)(x).reshape((-1,1))),
                                    function_library('negToPos')(function_library('scaledSine',p=p)(x).reshape((-1,1))),
                                    #function_library('scaledSine',p=p)(x),
                                    function_library('cubed')(function_library('scaledSine',p=p)(x).reshape((-1,1))),
                                    function_library('sine')(function_library('scaledSine',p=p)(x).reshape((-1,1))),
                                    function_library('exp')(function_library('scaledSine',p=p)(x).reshape((-1,1))),
                                    ]).T