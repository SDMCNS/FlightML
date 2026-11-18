/**
 * FlightML Operational Inefficiency & Navigation Infrastructure (TypeScript).
 */

export function keaHorizontalInefficiencyPct(
  actualFlownNm: number,
  greatCircleNm: number
): number {
  const gc = Math.max(1.0, greatCircleNm);
  const act = Math.max(gc, actualFlownNm);
  return ((act - gc) / gc) * 100.0;
}

export function kepPlannedInefficiencyPct(
  filedPlannedNm: number,
  greatCircleNm: number
): number {
  const gc = Math.max(1.0, greatCircleNm);
  const plan = Math.max(gc, filedPlannedNm);
  return ((plan - gc) / gc) * 100.0;
}

export function rlosRadioHorizonNm(
  stationElevationFt: number,
  aircraftAltitudeFt: number
): number {
  const hSt = Math.max(0.0, stationElevationFt);
  const hAc = Math.max(0.0, aircraftAltitudeFt);
  return 1.23 * (Math.sqrt(hSt) + Math.sqrt(hAc));
}

export function descentFlightPathAngleDeg(
  deltaAltitudeFt: number,
  distanceNm: number
): number {
  const dM = Math.max(0.01, distanceNm) * 1852.0;
  const dhM = deltaAltitudeFt * 0.3048;
  return (Math.atan2(dhM, dM) * 180.0) / Math.PI;
}

export function isCdoLevelOffSegment(
  deltaFl: number,
  durationS: number,
  flThreshold: number = 2.5,
  durationThresholdS: number = 150.0
): boolean {
  return Math.abs(deltaFl) <= flThreshold && durationS >= durationThresholdS;
}
