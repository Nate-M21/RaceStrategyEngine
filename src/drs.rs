use serde::{Deserialize, Serialize};

use crate::driver::Driver;

#[derive(Clone, Copy, Serialize, Deserialize, Debug, Default)]
pub struct DRSZone {
    pub drs_boost: f64,
    pub detection_point: f64,
    pub activation_point_start: f64,
    pub activation_point_end: f64,
}
#[derive(Clone, Default, Serialize, Deserialize, Debug)]
pub struct DRS {
    pub drs_zones: Vec<DRSZone>,
    pub delta_for_drs_activation: f64,
    pub drs_activation_lap: u8,
}

impl DRS {
    pub fn new(
        drs_zones: Vec<DRSZone>,
        delta_for_drs_activation: f64,
        drs_activation_lap: u8,
    ) -> Self {
        Self {
            drs_zones,
            delta_for_drs_activation,
            drs_activation_lap,
        }
    }
    pub fn get_drs_boost(&self, driver: &Driver) -> f64 {
        for zone in self.drs_zones.iter() {
            if zone.activation_point_start <= driver.get_current_lap_progress()
                && driver.get_current_lap_progress() <= zone.activation_point_end
            {
                return zone.drs_boost;
            }
        }

        return 0.0;
    }

    fn get_active_detection_point(&self, driver: &Driver) -> Option<f64> {
        let mut active_point = None;

        for zone in self.drs_zones.iter() {
            if driver.get_current_lap_progress() >= zone.detection_point {
                active_point = Some(zone.detection_point)
            }
        }

        return active_point;
    }

    pub fn progress_on_track_with_drs(&self, driver: &Driver, time_step: f64) -> f64 {
        let current_lap = driver.current_lap as usize;

        let lap_time = driver.get_driver_lap_time(current_lap);

        let drs_boost = self.get_drs_boost(driver);

        let drs_boosted_lap_time = lap_time - drs_boost;

        let progress_per_time_step = time_step / drs_boosted_lap_time;

        return progress_per_time_step;
    }

    pub fn driver_in_drs_activation_zone(&self, driver: &Driver) -> bool {
        for zone in self.drs_zones.iter() {
            if zone.detection_point == driver.drs_last_detection_point {
                if zone.activation_point_start <= driver.get_current_lap_progress()
                    && driver.get_current_lap_progress() <= zone.activation_point_end
                {
                    return true;
                }
            }
        }
        return false;
    }

    fn get_lap_progress_driver_ahead_at_detection_point(
        &self,
        driver_checking: &Driver,
        drivers: &[Driver],
    ) -> DriverAheadAtPoint {
        let mut drivers_ahead = Vec::with_capacity(22);

        for driver in drivers {
            let driver_lap_progress = driver.get_current_lap_progress();
            if driver.driver_position != driver_checking.driver_position
                && driver_lap_progress > driver_checking.drs_last_detection_point
                && driver_lap_progress > driver_checking.get_current_lap_progress()
            {
                drivers_ahead.push(driver)
            }
        }

        if drivers_ahead.is_empty() {
            return DriverAheadAtPoint::NoDriverAhead;
        } else {
            let mut closest_driver_lap_progress = drivers_ahead[0].get_current_lap_progress();
            for driver in drivers_ahead.iter() {
                let driver_lap_progress = driver.get_current_lap_progress();
                if driver_lap_progress < closest_driver_lap_progress {
                    closest_driver_lap_progress = driver_lap_progress
                }
            }
            return DriverAheadAtPoint::DriverAhead(closest_driver_lap_progress);
        }
    }

    pub fn driver_drs_eligibilty_at_detection(
        &self,
        driver: &Driver,
        drivers: &[Driver],
    ) -> DrsEligibilty {
        let current_lap = driver.current_lap as usize;
        let active_detection_point = self.get_active_detection_point(driver);
        let drs_is_available = current_lap as u8 >= self.drs_activation_lap;

        let active_detection_point = match active_detection_point {
            Some(point) => point,
            None => return DrsEligibilty::NotAtDetection,
        };

        if !drs_is_available {
            let drs_eligibilty = false;
            return DrsEligibilty::AtDetectionPoint(active_detection_point, drs_eligibilty);
        }

        let driver_ahead_lap_progress =
            match self.get_lap_progress_driver_ahead_at_detection_point(driver, drivers) {
                DriverAheadAtPoint::DriverAhead(position) => position,
                DriverAheadAtPoint::NoDriverAhead => {
                    let drs_eligibilty = false;
                    return DrsEligibilty::AtDetectionPoint(active_detection_point, drs_eligibilty);
                }
            };

        let gap = driver_ahead_lap_progress - driver.get_current_lap_progress();
        // TODO use reference time for everyone from leader
        let time_delta = gap * driver.get_driver_lap_time(current_lap);

        let drs_eligibilty = time_delta <= self.delta_for_drs_activation;
        return DrsEligibilty::AtDetectionPoint(active_detection_point, drs_eligibilty);
    }

    pub fn checked_driver_drs_status_at_point(
        &self,
        driver: &mut Driver,
        detection_point: f64,
    ) -> bool {
        let current_lap = driver.current_lap;

        if current_lap == driver.drs_last_lap_checked
            && detection_point == driver.drs_last_detection_point
        {
            return true;
        }

        driver.drs_last_detection_point = detection_point;
        driver.drs_last_lap_checked = current_lap;

        return false;
    }
}

pub enum DrsEligibilty {
    NotAtDetection,
    AtDetectionPoint(f64, bool),
}

pub enum DriverAheadAtPoint {
    DriverAhead(f64),
    NoDriverAhead,
}
