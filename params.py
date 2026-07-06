import numpy as np
from datetime import datetime


SOLAR_CONSTANT  = 1377 * 0.967   
LIGHT_SPEED     = 299_792_458    
ECLIPSE_ANGLE   = np.deg2rad(65.5)  



PHI_EARTH_2000  = np.deg2rad(279 + 44/60)  
THETA_G0        = 0.0                       
OMEGA_E         = 7.29e-5                   


G01 = -29_496.6   
G11 =  -1_586.3   
H11 =   4_944.0   

H0       = np.linalg.norm([G01, G11, H11]) 
DIP_COEL = np.arccos(G01 / H0)              
DIP_LONG = np.arctan2(H11, G11)             

DATE        = [2019, 6, 1, 0, 0, 0]    
T0_JD       = 2_456_738.0              

def _decimal_year(date: list) -> float:
    """Convierte una fecha [y,m,d,h,min,s] a año decimal."""
    dt    = datetime(*date)
    start = datetime(date[0], 1, 1)
    end   = datetime(date[0] + 1, 1, 1)
    return date[0] + (dt - start).total_seconds() / (end - start).total_seconds()

DEC_YEAR       = _decimal_year(DATE)
_DT            = datetime(*DATE)
_DATE2000      = datetime(2000, 1, 1, 12, 0, 0)
SEC_FROM_2000  = (_DT - _DATE2000).total_seconds()  


G_CONST    = 6.67e-11         
M_EARTH    = 5.97e24          


MU_EARTH   = 3.986004418e14    
R_EARTH    = 6_378_137.0       

A_ORB      = 6_978_137.0              
ECC        = 0.0                      
INC        = np.deg2rad(98.72636)     
RAAN0      = np.deg2rad(0.0)          
RAAN_RATE  = 2.02e-7                  
AOP        = np.deg2rad(0.0)          
TA0        = np.deg2rad(0.0)          


J2_RE2 = -2 * RAAN_RATE * A_ORB**(3.5) * (1 - ECC**2)**2 / (3 * np.sqrt(MU_EARTH) * np.cos(INC))

N_ORB      = np.sqrt(MU_EARTH / A_ORB**3)  
T_ORB      = 2 * np.pi / N_ORB            

QUAT_0 = np.array([-0.129409522551260,
                    0.224143868042013,
                    0.482962913144534,
                    0.836516303737808])

OMEGA_BI_B0 = np.array([0.1, -0.1, -0.1])


C_ADCS = np.array([
    [ 0,  0, -1],
    [ 0, -1,  0],
    [-1,  0,  0]
], dtype=float)


_I_CATIA = np.array([
    [ 3.129, -0.004, -0.007],
    [-0.004,  2.418, -0.008],
    [-0.007, -0.008,  2.536]
])


INERTIA = C_ADCS @ _I_CATIA @ C_ADCS.T

MASS    = 50.0   

R_CM_B  = C_ADCS @ np.array([0.030, -0.002, 0.211])

DIM = np.array([0.25, -0.25, 0.25, -0.25, 0.30, -0.30])

_S2P = (DIM[2] - DIM[3]) * (DIM[4] - DIM[5]) 


_A_CATIA = np.array([
    (DIM[2]-DIM[3])*(DIM[4]-DIM[5]) + 2*_S2P,
    (DIM[2]-DIM[3])*(DIM[4]-DIM[5]),
    (DIM[0]-DIM[1])*(DIM[4]-DIM[5]),
    (DIM[0]-DIM[1])*(DIM[4]-DIM[5]),
    (DIM[2]-DIM[3])*(DIM[0]-DIM[1]),
    (DIM[2]-DIM[3])*(DIM[0]-DIM[1]),
])
A_FACES = _A_CATIA[::-1]  


N_FACES = np.array([
    [ 1,  0,  0],
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
    [ 0,  0, -1]
], dtype=float)


_MCP_CATIA = np.array([
    [(DIM[0]+DIM[0])/2, (DIM[2]+DIM[3])/2, (DIM[4]+DIM[5])/2],
    [(DIM[1]+DIM[1])/2, (DIM[2]+DIM[3])/2, (DIM[4]+DIM[5])/2],
    [(DIM[0]+DIM[1])/2, (DIM[2]+DIM[2])/2, (DIM[4]+DIM[5])/2],
    [(DIM[0]+DIM[1])/2, (DIM[3]+DIM[3])/2, (DIM[4]+DIM[5])/2],
    [(DIM[0]+DIM[1])/2, (DIM[2]+DIM[3])/2, (DIM[4]+DIM[4])/2],
    [(DIM[0]+DIM[1])/2, (DIM[2]+DIM[3])/2, (DIM[5]+DIM[5])/2],
])
MCP_B = (C_ADCS @ (_MCP_CATIA + np.array([0, 0, DIM[4]])).T).T


CM_MM01 = np.array([
    [-27723,   1088,   -239],
    [  -764, -25026,   -219],
    [  1299,   -416, -23472]
], dtype=float)
CO_MM01 = np.array([2.5332, 2.5200, 2.4915])


CM_MM02 = np.array([
    [-27799,     42,     95],
    [  -1261, -26308,   -43],
    [   2693,    458, -24088]
], dtype=float)
CO_MM02 = np.array([2.5348, 2.5260, 2.4664])


CM_MM03 = np.array([
    [9988,    0,     0],
    [   0, 9989,     0],
    [   0,    0, 10036]
], dtype=float)
CO_MM03 = np.array([0.0008, -0.0006, 0.0007])


MM_WORKING = np.array([1, 1, 0])
M_MAX_MTQ = 15.0   

OMEGA_CMD = np.array([0.0, 0.0, 0.1])  

MT_WORKING = np.array([1, 1, 1])


K_PB = 2   
K_PE = 8   

DELTA = 1e-10  


EPSILON_AERO = 0.1   
ETA_AERO     = 0.1   


AIR_DENSITY = np.array([
    1.2250e+00, 3.8990e-02, 1.7740e-02, 8.2790e-03,
    3.9720e-03, 1.9950e-03, 1.0570e-03, 5.8210e-04,
    3.2060e-04, 1.7180e-04, 8.7700e-05, 4.1780e-05,
    1.9050e-05, 8.3370e-06, 3.3960e-06, 1.3430e-06,
    5.2970e-07, 9.6610e-08, 2.4380e-08, 8.4840e-09,
    3.8450e-09, 2.0700e-09, 1.2440e-09, 5.4640e-10,
    2.7890e-10, 7.2480e-11, 2.4180e-11, 9.1580e-12,
    3.7250e-12, 1.5850e-12, 6.9670e-13, 1.4540e-13,
    3.6140e-14, 1.1700e-14, 5.2450e-15, 3.0190e-15,
])

ALTITUDE_TABLE = np.array([
         0,  25000,  30000,  35000,  40000,  45000,  50000,  55000,
     60000,  65000,  70000,  75000,  80000,  85000,  90000,  95000,
    100000, 110000, 120000, 130000, 140000, 150000, 160000, 180000,
    200000, 250000, 300000, 350000, 400000, 450000, 500000, 600000,
    700000, 800000, 900000, 1000000,
], dtype=float)


C_SPEC   = np.array([0.29, 0.29, 0.29, 0.29, 0.29, 0.0727])
C_DIFFUS = np.array([0.29, 0.29, 0.29, 0.29, 0.29, 0.007 ])


RMD_B = np.array([0.0, 0.0, 1.0])