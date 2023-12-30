import pandas as pd
from geopy.distance import geodesic



def estimate(A, coordinate, data, month):
    '''E(kWh) = A(Total solar panel Area-Total roof area for us-) * r(solar panel yield or efficiency) * H (Average solar radiation on tilted panels[kWh/m^2]) * Pr(Performance Ratio)
    A will come from satellite images segmentation
    r is constant value (Solar panel efficiency is generally around 15-20%-We say 17.5)
    PR = Performance ratio, coefficient for losses (range between 0.5 and 0.9, default value = 0.75)
    month = If you want to learn estimating for a selected month then type months name; if you want to annual estimate then type 'Year'
    '''
    r = 0.175
    PR = 0.75
    
    distances = {col: calculate_distance(coordinate, col) for col in data.columns}
    nearestCoordinate = min(distances, key=distances.get)
    if month == "Year":
        H = data.loc["Year",nearestCoordinate] * 0.277778
        energy = A * r * H * PR
        return energy
    H = data.loc[month,nearestCoordinate] * 0.277778
    energy = A * r * H * PR
    return energy

def calculate_distance(coord1, coord2):
    lat1, lon1 = map(float, coord1.split('x'))
    lat2, lon2 = map(float, coord2.split('x'))
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

data = pd.read_csv("SolarEnergySystem/ankaraRadiation.csv")
print(estimate(5,"39.9309x32.8599",data,"Year"))






