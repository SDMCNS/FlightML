/**
 * FlightML Climate & Emissions Microphysics (TypeScript).
 */

export const EI_CO2_KG_PER_KG_FUEL = 3.155;
export const EI_H2O_KG_PER_KG_FUEL = 1.230;
export const EI_NOX_G_PER_KG_FUEL_CRUISE = 13.5;

export function sonntagSaturationVaporPressureLiquidHpa(temperatureK: number): number {
  const t = Math.max(170.0, Math.min(330.0, temperatureK));
  const lnE =
    -6096.9385 / t +
    16.635794 -
    0.02711193 * t +
    1.673952e-5 * t ** 2 +
    2.433502 * Math.log(t);
  return Math.exp(lnE);
}

export function sonntagSaturationVaporPressureIceHpa(temperatureK: number): number {
  const t = Math.max(170.0, Math.min(273.15, temperatureK));
  const lnE =
    -6024.5282 / t +
    24.7219 -
    0.010613868 * t +
    1.3198825e-5 * t ** 2 -
    0.49382577 * Math.log(t);
  return Math.exp(lnE);
}

export function relativeHumidityOverIcePct(rhLiquidPct: number, temperatureK: number): number {
  const t = Math.max(170.0, temperatureK);
  if (t >= 273.15) return rhLiquidPct;
  const eLiq = sonntagSaturationVaporPressureLiquidHpa(t);
  const eIce = sonntagSaturationVaporPressureIceHpa(t);
  return rhLiquidPct * (eLiq / Math.max(1e-9, eIce));
}

export function schmidtApplemanThresholdC(
  pressureHpa: number,
  propulsionEfficiency: number = 0.35
): number {
  const p = Math.max(50.0, Math.min(1000.0, pressureHpa));
  const eta = Math.max(0.2, Math.min(0.5, propulsionEfficiency));
  const pRatio = Math.max(0.15, Math.min(1.0, p / 250.0));
  return -41.5 + 2.8 * Math.log(pRatio) + (eta - 0.33) * 12.0;
}

export function isPersistentContrailForming(
  temperatureC: number,
  pressureHpa: number,
  rhIcePct: number,
  propulsionEfficiency: number = 0.35
): boolean {
  const tSac = schmidtApplemanThresholdC(pressureHpa, propulsionEfficiency);
  return temperatureC <= tSac && rhIcePct >= 100.0;
}

export function altitudeFuelBurnPenalty(
  flownFl: number,
  optimalFl: number,
  pressureRatio: number = 30.0
): number {
  if (flownFl <= 0 || optimalFl <= 0) return 0.0;
  const deltaFl = flownFl - optimalFl;
  if (deltaFl < 0) {
    const deficitThousandsFt = Math.abs(deltaFl) / 10.0;
    const prSensitivity = 1.0 + Math.max(0.0, pressureRatio - 25.0) / 100.0;
    return 0.015 * deficitThousandsFt * prSensitivity;
  } else if (deltaFl > 0) {
    const excessThousandsFt = deltaFl / 10.0;
    return 0.02 * excessThousandsFt;
  }
  return 0.0;
}

export function fuelToCo2Kg(fuelKg: number): number {
  return Math.max(0.0, fuelKg) * EI_CO2_KG_PER_KG_FUEL;
}

export function fuelToNoxG(fuelKg: number, cruiseEiNox: number = EI_NOX_G_PER_KG_FUEL_CRUISE): number {
  return Math.max(0.0, fuelKg) * cruiseEiNox;
}

export function contrailEnergyForcingRfJ(
  distanceKm: number,
  opticalDepth: number = 0.3,
  widthKm: number = 10.0,
  durationHours: number = 4.0,
  instantaneousRfWM2: number = 12.0
): number {
  const distM = Math.max(0.0, distanceKm) * 1000.0;
  const widthM = Math.max(100.0, widthKm) * 1000.0;
  const durationS = Math.max(60.0, durationHours) * 3600.0;
  const areaM2 = distM * widthM;
  return instantaneousRfWM2 * areaM2 * durationS * (opticalDepth / 0.3);
}

export function sootNumberToOpticalDepth(
  eiNumPerKg: number,
  nominalEiNum: number = 1.0e15,
  nominalOpticalDepth: number = 0.30
): number {
  if (eiNumPerKg <= 0.0) return 0.05;
  const ratio = Math.max(1.0e-3, Math.min(100.0, eiNumPerKg / nominalEiNum));
  const tau = nominalOpticalDepth * Math.pow(ratio, 0.65);
  return Number(Math.max(0.02, Math.min(1.5, tau)).toFixed(4));
}

export function initialIceCrystalConcentration(
  eiNumPerKg: number,
  fuelFlowKgS: number,
  wingspanM: number,
  trueAirspeedMps: number
): number {
  if (wingspanM <= 0.0 || trueAirspeedMps <= 0.0 || fuelFlowKgS <= 0.0) return 0.0;
  const aWake = (Math.PI / 4.0) * Math.pow(wingspanM, 2);
  const volRateM3S = aWake * trueAirspeedMps;
  const crystalsPerS = eiNumPerKg * fuelFlowKgS;
  return crystalsPerS / volRateM3S;
}
