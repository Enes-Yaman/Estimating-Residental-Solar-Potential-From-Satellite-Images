import pandas as pd
from geopy.distance import geodesic
import math

def estimate(A, coordinate, month, degree = 27):
    data = pd.read_csv(r"\SolarEnergySystem\ankaraRadiation.csv")
    '''E(kWh) = A(Total solar panel Area-Total roof area for us-) * r(solar panel yield or efficiency) * H (Average solar radiation on tilted panels[kWh/m^2]) * Pr(Performance Ratio)
    A will come from satellite images segmentation (Direction and Area)
    r is constant value (Solar panel efficiency is generally around 15-20%-We say 17.5)
    H will come from 'ankaraRadiation.csv' dataset. This dataset includes informations in mj unit. mj * 0.277778 = kWh/m^2
    PR = Performance ratio, coefficient for losses (range between 0.5 and 0.9, default value = 0.75)
    month = If you want to learn estimating for a selected month then type months name; if you want to annual estimate then type 'Year'
    Degree is the degree between surface and rooftop. Enter a degree or type 'Default'
    '''
    r = 0.175
    PR = 0.75
    flat = A.get('F')
    N = A.get("N")
    S = A.get("S")
    E = A.get("E")
    W = A.get("W")
    distances = {col: calculate_distance(coordinate, col) for col in data.columns}
    nearestCoordinate = min(distances, key=distances.get)
    if month == "Year":
        H = data.loc["Year",nearestCoordinate] * 0.277778
        flatEnergy = flat * r * H * PR
        Nenergy = N * r * H * PR * abs(math.sin(degree))
        Senergy = S * r * H * PR * abs(math.sin(degree))
        Eenergy = E * r * H * PR * abs(math.sin(degree))
        Wenergy = W * r * H * PR * abs(math.sin(degree))
        return f"Estimated energy: {flatEnergy} kWh for flat surfaces , {Nenergy} kWh for N, {Senergy} kWh for S, {Eenergy} kWh for E, {Wenergy} kWh for W, total {flatEnergy+Nenergy+Senergy+Eenergy+Wenergy} per Year"
    H = data.loc[month,nearestCoordinate] * 0.277778
    flatEnergy = flat * r * H * PR
    Nenergy = N * r * H * PR * abs(math.sin(degree))
    Senergy = S * r * H * PR * abs(math.sin(degree))
    Eenergy = E * r * H * PR * abs(math.sin(degree))
    Wenergy = W * r * H * PR * abs(math.sin(degree))
    return f"Estimated energy: {flatEnergy} kWh for flat surfaces , {Nenergy} kWh for N, {Senergy} kWh for S, {Eenergy} kWh for E, {Wenergy} kWh for W, total {flatEnergy+Nenergy+Senergy+Eenergy+Wenergy} for {month}"

def calculate_distance(coord1, coord2):
    lat1, lon1 = map(float, coord1.split('x'))
    lat2, lon2 = map(float, coord2.split('x'))
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers


