/**
 * FlightML Aerodynamics & Kinematics (TypeScript).
 */

import { GAMMA_AIR, R_GAS_AIR, MPS_TO_KTS, G_GRAVITY } from './atmosphere';

const KTS_TO_MPS = 1.0 / MPS_TO_KTS;

export function machToTasKts(mach: number, temperatureK: number): number {
  const m = Math.max(0.0, mach);
  const t = Math.max(150.0, temperatureK);
  const speedOfSoundMps = Math.sqrt(GAMMA_AIR * R_GAS_AIR * t);
  return m * speedOfSoundMps * MPS_TO_KTS;
}

export function tasToMach(tasKts: number, temperatureK: number): number {
  const v = Math.max(0.0, tasKts);
  const t = Math.max(150.0, temperatureK);
  const speedOfSoundKts = Math.sqrt(GAMMA_AIR * R_GAS_AIR * t) * MPS_TO_KTS;
  return v / Math.max(1e-6, speedOfSoundKts);
}

export function headwindTailwindComponentKts(
  trackDeg: number,
  windFromDeg: number,
  windSpeedKts: number
): number {
  const angleRad = ((windFromDeg - trackDeg - 180.0) * Math.PI) / 180.0;
  return windSpeedKts * Math.cos(angleRad);
}

export function crosswindComponentKts(
  headingOrRunwayDeg: number,
  windFromDeg: number,
  windSpeedKts: number
): number {
  const angleRad = ((windFromDeg - headingOrRunwayDeg) * Math.PI) / 180.0;
  return Math.abs(windSpeedKts * Math.sin(angleRad));
}

export function equivalentStillAirDistanceNm(
  groundDistNm: number,
  tasKts: number,
  groundspeedKts: number
): number {
  const gs = Math.max(10.0, groundspeedKts);
  const tas = Math.max(10.0, tasKts);
  return (tas / gs) * Math.max(0.0, groundDistNm);
}

export function turnRadiusNm(tasKts: number, bankAngleDeg: number): number {
  const tasMps = tasKts * KTS_TO_MPS;
  const phiRad = (Math.min(80.0, Math.max(1.0, Math.abs(bankAngleDeg))) * Math.PI) / 180.0;
  const radiusM = Math.pow(tasMps, 2) / (G_GRAVITY * Math.tan(phiRad));
  return radiusM / 1852.0;
}

export function loadFactorG(bankAngleDeg: number): number {
  const phiRad = (Math.min(85.0, Math.max(0.0, Math.abs(bankAngleDeg))) * Math.PI) / 180.0;
  return 1.0 / Math.cos(phiRad);
}
