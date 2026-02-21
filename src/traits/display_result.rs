use crate::race_simulation::DriverResult;

pub trait DisplayResult {
    fn display_results(&self, simulation_result: &Vec<DriverResult>) {
        let mut drivers: Vec<&DriverResult> = simulation_result.iter().collect();

        drivers.sort_by(|a, b| a.driver_race_time.total_cmp(&b.driver_race_time));
        println!("{}", "*#".repeat(50));
        println!("Final Grid");
        println!("{}", "-".repeat(50));
        for driver in drivers {
            println!(
                "P{}\t{} \t\t{}",
                driver.driver_position,
                driver.name.chars().take(3).collect::<String>(),
                driver.driver_race_time,
            )
        }
        println!("{}", "*#".repeat(50));
    }
}
