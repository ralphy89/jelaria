"""Associate license plate detections with tracked vehicles."""

def assign_plate_to_vehicle(license_plate, vehicle_tracks):
    """
    Find the vehicle that fully contains the license plate bounding box.

    A plate is assigned only when its box lies entirely inside a vehicle box.
    Plates that cannot be matched are ignored (car_id = -1).

    Args:
        license_plate: [x1, y1, x2, y2, score, class_id]
        vehicle_tracks: SORT output — array of [x1, y1, x2, y2, car_id]

    Returns:
        tuple: (xcar1, ycar1, xcar2, ycar2, car_id) or (-1, -1, -1, -1, -1)
    """
    x1, y1, x2, y2, score, class_id = license_plate

    for track in vehicle_tracks:
        xcar1, ycar1, xcar2, ycar2, car_id = track

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id

    return -1, -1, -1, -1, -1
