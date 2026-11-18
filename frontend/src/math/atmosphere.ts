/**
 * FlightML International Standard Atmosphere (ISA) Physics (TypeScript).
 */

export const T0_K = 288.15;
export const P0_HPA = 1013.25;
export const LAPSE_RATE_K_PER_M = 0.0065;
export const TROPOPAUSE_ALT_M = 11000.0;
export const T_TROPOPAUSE_K = 216.65;
export const R_GAS_AIR = 287.05287;
export const GAMMA_AIR = 1.4;
export const G_GRAVITY = 9.80665;
export const M_PER_FT = 0.3048;
export const FT_PER_M = 1.0 / 0.3048;
export const MPS_TO_KTS = 1.94384449;

export function flightLevelToAltitudeM(flightLevel: number): number {
  return flightLevel * 100.0 * M_PER_FT;
}

export function flightLevelToAltitudeFt(flightLevel: number): number {
  return flightLevel * 100.0;
}

export function altitudeMToFlightLevel(altM: number): number {
  return (altM * FT_PER_M) / 100.0;
}

export function isaTemperatureK(altitudeM: number): number {
  const h = Math.max(0.0, altitudeM);
  if (h <= TROPOPAUSE_ALT_M) {
    return T0_K - LAPSE_RATE_K_PER_M * h;
  }
  return T_TROPOPAUSE_K;
}

export function isaTemperatureC(altitudeM: number): number {
  return isaTemperatureK(altitudeM) - 273.15;
}

export function isaPressureHpa(altitudeM: number): number {
  const h = Math.max(0.0, altitudeM);
  const exponentTropo = G_GRAVITY / (R_GAS_AIR * LAPSE_RATE_K_PER_M); // ~5.25588

  if (h <= TROPOPAUSE_ALT_M) {
    const tempRatio = 1.0 - (LAPSE_RATE_K_PER_M * h) / T0_K;
    return P0_HPA * Math.pow(tempRatio, exponentTropo);
  } else {
    const pTropo = P0_HPA * Math.pow(T_TROPOPAUSE_K / T0_K, exponentTropo);
    const deltaH = h - TROPOPAUSE_ALT_M;
    return pTropo * Math.exp(-(G_GRAVITY * deltaH) / (R_GAS_AIR * T_TROPOPAUSE_K));
  }
}

export function isaDensityKgM3(altitudeM: number): number {
  const pPa = isaPressureHpa(altitudeM) * 100.0;
  const tK = isaTemperatureK(altitudeM);
  return pPa / (R_GAS_AIR * tK);
}

export function speedOfSoundMps(altitudeM: number): number {
  const tK = isaTemperatureK(altitudeM);
  return Math.sqrt(GAMMA_AIR * R_GAS_AIR * tK);
}

export function speedOfSoundKts(altitudeM: number): number {
  return speedOfSoundMps(altitudeM) * MPS_TO_KTS;
}
