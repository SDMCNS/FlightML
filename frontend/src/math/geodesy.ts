/**
 * FlightML Mathematical Geodesy & Spherical Trigonometry (TypeScript).
 */

export const EARTH_RADIUS_NM = 3440.065;
export const EARTH_RADIUS_KM = 6371.0088;

export function toRadians(deg: number): number {
  return (deg * Math.PI) / 180.0;
}

export function toDegrees(rad: number): number {
  return (rad * 180.0) / Math.PI;
}

export function normalizeLongitudeDeg(lon: number): number {
  const wrapped = ((lon + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
  if (wrapped === -180.0 && lon > 0) return 180.0;
  return wrapped;
}

export function angularDifferenceDeg(lon1: number, lon2: number): number {
  return (((lon2 - lon1 + 180.0) % 360.0 + 360.0) % 360.0) - 180.0;
}

export function haversineDistanceNm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const phi1 = toRadians(lat1);
  const phi2 = toRadians(lat2);
  const dPhi = toRadians(lat2 - lat1);
  const dLambda = toRadians(lon2 - lon1);

  const a =
    Math.sin(dPhi / 2.0) ** 2 +
    Math.cos(phi1) * Math.cos(phi2) * Math.sin(dLambda / 2.0) ** 2;
  const c = 2.0 * Math.atan2(Math.sqrt(Math.max(0.0, a)), Math.sqrt(Math.max(0.0, 1.0 - a)));
  return EARTH_RADIUS_NM * c;
}

export function haversineDistanceKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  return haversineDistanceNm(lat1, lon1, lat2, lon2) * (EARTH_RADIUS_KM / EARTH_RADIUS_NM);
}

export function initialBearingDeg(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const phi1 = toRadians(lat1);
  const phi2 = toRadians(lat2);
  const dLambda = toRadians(lon2 - lon1);

  const y = Math.sin(dLambda) * Math.cos(phi2);
  const x =
    Math.cos(phi1) * Math.sin(phi2) -
    Math.sin(phi1) * Math.cos(phi2) * Math.cos(dLambda);

  const bearing = toDegrees(Math.atan2(y, x));
  return (bearing + 360.0) % 360.0;
}

export function finalBearingDeg(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const b = initialBearingDeg(lat2, lon2, lat1, lon1);
  return (b + 180.0) % 360.0;
}

export function crossTrackDistanceNm(
  pLat: number,
  pLon: number,
  startLat: number,
  startLon: number,
  endLat: number,
  endLon: number
): number {
  const d13 = haversineDistanceNm(startLat, startLon, pLat, pLon);
  if (d13 < 1e-6) return 0.0;

  const delta13 = d13 / EARTH_RADIUS_NM;
  const theta13 = toRadians(initialBearingDeg(startLat, startLon, pLat, pLon));
  const theta12 = toRadians(initialBearingDeg(startLat, startLon, endLat, endLon));

  const xtdRad = Math.asin(Math.max(-1.0, Math.min(1.0, Math.sin(delta13) * Math.sin(theta13 - theta12))));
  return xtdRad * EARTH_RADIUS_NM;
}

export function sphericalMidpoint(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): [number, number] {
  const phi1 = toRadians(lat1);
  const lambda1 = toRadians(lon1);
  const phi2 = toRadians(lat2);
  const lambda2 = toRadians(lon2);

  const dLambda = lambda2 - lambda1;
  const bx = Math.cos(phi2) * Math.cos(dLambda);
  const by = Math.cos(phi2) * Math.sin(dLambda);

  const phi3 = Math.atan2(
    Math.sin(phi1) + Math.sin(phi2),
    Math.sqrt((Math.cos(phi1) + bx) ** 2 + by ** 2)
  );
  const lambda3 = lambda1 + Math.atan2(by, Math.cos(phi1) + bx);

  const midLat = toDegrees(phi3);
  const midLon = ((toDegrees(lambda3) + 540.0) % 360.0) - 180.0;
  return [midLat, midLon];
}

export function circularMeanLongitude(lons: number[]): number {
  if (lons.length === 0) return 0.0;
  let sinSum = 0.0;
  let cosSum = 0.0;
  for (const lon of lons) {
    const rad = toRadians(lon);
    sinSum += Math.sin(rad);
    cosSum += Math.cos(rad);
  }
  if (Math.abs(sinSum) < 1e-12 && Math.abs(cosSum) < 1e-12) return 0.0;
  return toDegrees(Math.atan2(sinSum, cosSum));
}

export function splitPolylineAtAntimeridian(
  coords: [number, number][],
  thresholdDeg: number = 180.0
): [number, number][][] {
  if (coords.length === 0) return [];
  const segments: [number, number][][] = [];
  let currentSegment: [number, number][] = [coords[0]];

  for (let i = 1; i < coords.length; i++) {
    const lon1 = coords[i - 1][1];
    const lon2 = coords[i][1];
    if (Math.abs(lon2 - lon1) > thresholdDeg) {
      segments.push(currentSegment);
      currentSegment = [coords[i]];
    } else {
      currentSegment.push(coords[i]);
    }
  }
  if (currentSegment.length > 0) {
    segments.push(currentSegment);
  }
  return segments;
}
