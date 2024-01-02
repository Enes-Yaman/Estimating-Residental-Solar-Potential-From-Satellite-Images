import math

import pandas as pd
from geopy.distance import geodesic


def estimate(A, coordinate, month='Year', degree=27):
    data = pd.read_csv(r"ankaraRadiation.csv")
    '''E(kWh) = A(Total solar panel Area-Total roof area for us-) * r(solar panel yield or efficiency) * H (Average solar radiation on tilted panels[kWh/m^2]) * Pr(Performance Ratio)
    A will come from satellite images segmentation (Direction and Area)
    r is constant value (Solar panel efficiency is generally around 15-20%-We say 17.5)
    H will come from 'ankaraRadiation.csv' dataset. This dataset includes informations in mj unit. mj * 0.277778 = kWh/m^2
    PR = Performance ratio, coefficient for losses (range between 0.5 and 0.9, default value = 0.75)
    month = If you want to learn estimating for a selected month then type months name; if you want to annual estimate then type 'Year'
    Degree is the degree between surface and rooftop. Enter a degree or type 'Default'
    '''
    degree = math.radians(degree)
    r = 0.175
    PR = 0.75
    flat = A.get('F')
    north = A.get("N")
    south = A.get("S")
    east = A.get("E")
    west = A.get("W")
    distances = {col: calculate_distance(coordinate, col) for col in data.columns}
    nearestCoordinate = min(distances, key=distances.get)
    if month == "Year":
        H = data.loc["Year", nearestCoordinate] * 0.277778
    else:
        H = data.loc[month, nearestCoordinate] * 0.277778
    flat_Energy = round(flat * r * H * PR, 2)
    north_energy = round(north * r * H * PR * abs(math.cos(degree)), 2)
    south_energy = round(south * r * H * PR * abs(math.cos(degree)), 2)
    east_energy = round(east * r * H * PR * abs(math.cos(degree)), 2)
    west_energy = round(west * r * H * PR * abs(math.cos(degree)), 2)
    return (f"Estimated energy: {flat_Energy} kWh for flat surfaces , {north_energy} kWh for north, {south_energy} kWh for south, {east_energy} kWh for east, {west_energy} kWh for west, total {flat_Energy + north_energy + south_energy + east_energy + west_energy}"
            f"{' per year' if month == 'Year' else f' for {month}'}")


def calculate_distance(coord1, coord2):
    lat1, lon1 = map(float, coord1.split('x'))
    lat2, lon2 = map(float, coord2.split('x'))
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers
