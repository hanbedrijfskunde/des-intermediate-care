import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime, timedelta

class IntermediateCareSimulation:
    """
    Discrete Event Simulation model for intermediate care facilities
    """
    def __init__(self, 
                 regular_beds=50,       # Number of regular nursing/ELV beds
                 crisis_beds=10,        # Number of crisis beds
                 regular_arrival_rate=8,   # Regular patients per day
                 crisis_arrival_rate=2,    # Crisis patients per day
                 regular_los_mean=14,     # Mean length of stay (days) for regular patients
                 crisis_los_mean=7,       # Mean length of stay (days) for crisis patients
                 triage_delay_mean=1.5,   # Mean triage/transfer delay (days)
                 time_dependent_arrivals=False,  # Whether to use time-dependent arrival rates
                 weekend_admission_restriction=True,  # Restrict admissions on weekends
                 night_admission_restriction=True,   # Restrict admissions at night
                 simulation_duration=365,  # Simulation duration (days)
                 warm_up_period=30):      # Warm-up period (days)
        
        self.regular_beds = regular_beds
        self.crisis_beds = crisis_beds
        self.regular_arrival_rate = regular_arrival_rate
        self.crisis_arrival_rate = crisis_arrival_rate
        self.regular_los_mean = regular_los_mean
        self.crisis_los_mean = crisis_los_mean
        self.triage_delay_mean = triage_delay_mean
        self.time_dependent_arrivals = time_dependent_arrivals
        self.weekend_admission_restriction = weekend_admission_restriction
        self.night_admission_restriction = night_admission_restriction
        self.simulation_duration = simulation_duration
        self.warm_up_period = warm_up_period
        
        # Initialize simulation environment and resources
        self.env = simpy.Environment()
        self.regular_bed_resource = simpy.Resource(self.env, capacity=regular_beds)
        self.crisis_bed_resource = simpy.Resource(self.env, capacity=crisis_beds)
        
        # Initialize statistics collection
        self.stats = {
            'regular_waiting_times': [],
            'crisis_waiting_times': [],
            'regular_queue_length': defaultdict(int),
            'crisis_queue_length': defaultdict(int),
            'regular_occupancy': defaultdict(int),
            'crisis_occupancy': defaultdict(int),
            'regular_daily_arrivals': defaultdict(int),
            'crisis_daily_arrivals': defaultdict(int),
            'daily_discharges': defaultdict(int),
            'denied_admissions': 0,
            'patient_log': []
        }
        
        # Track the time of the last statistics collection for queue lengths
        self.last_stats_time = 0

    def is_admission_allowed(self, time):
        """Check if admissions are allowed at the given time based on time restrictions"""
        day_of_week = int((time // 24) % 7)  # 0 = Monday, 6 = Sunday
        hour_of_day = time % 24
        
        # Check weekend restriction
        if self.weekend_admission_restriction and day_of_week >= 5:  # Weekend (Saturday and Sunday)
            return False
        
        # Check night-time restriction
        if self.night_admission_restriction and (hour_of_day < 8 or hour_of_day >= 17):  # Outside 8 AM to 5 PM
            return False
        
        return True

    def get_arrival_rate(self, time, is_crisis):
        """Get the arrival rate based on time of day and day of week if time-dependent arrivals are enabled"""
        if not self.time_dependent_arrivals:
            return self.crisis_arrival_rate if is_crisis else self.regular_arrival_rate
        
        # Extract day and hour information
        day_of_week = int((time // 24) % 7)  # 0 = Monday, 6 = Sunday
        hour_of_day = int(time % 24)
        
        base_rate = self.crisis_arrival_rate if is_crisis else self.regular_arrival_rate
        
        # Define multipliers for time-dependent variations
        day_multiplier = 1.0
        if day_of_week < 5:  # Weekday
            day_multiplier = 1.2
        else:  # Weekend
            day_multiplier = 0.8
        
        hour_multiplier = 1.0
        if 8 <= hour_of_day < 17:  # Daytime (8 AM to 5 PM)
            hour_multiplier = 1.5
        elif 17 <= hour_of_day < 22:  # Evening (5 PM to 10 PM)
            hour_multiplier = 1.2
        else:  # Night (10 PM to 8 AM)
            hour_multiplier = 0.5
        
        return base_rate * day_multiplier * hour_multiplier / 24  # Convert daily rate to hourly

    def patient_generator(self, is_crisis=False):
        """Generate patient arrivals according to a Poisson process"""
        patient_id = 0
        
        while True:
            # Calculate interarrival time based on Poisson process
            # Convert daily rate to hourly rate
            hourly_rate = self.get_arrival_rate(self.env.now, is_crisis)
            interarrival_time = np.random.exponential(1.0 / hourly_rate)
            
            # Wait until next arrival
            yield self.env.timeout(interarrival_time)
            
            # Create a new patient
            patient_id += 1
            arrival_time = self.env.now
            patient_type = "Crisis" if is_crisis else "Regular"
            
            # Log the patient arrival
            if self.env.now >= self.warm_up_period * 24:  # Only record statistics after warm-up period
                day = int(self.env.now // 24)
                if is_crisis:
                    self.stats['crisis_daily_arrivals'][day] += 1
                else:
                    self.stats['regular_daily_arrivals'][day] += 1
            
            # Start the patient process
            self.env.process(self.patient_process(patient_id, arrival_time, is_crisis))

    def patient_process(self, patient_id, arrival_time, is_crisis):
        """Handle the patient journey through the intermediate care system"""
        # Determine if admission is allowed at arrival time
        admission_allowed = self.is_admission_allowed(self.env.now)
        
        # If admissions are restricted at this time, patient must wait until next allowed time
        if not admission_allowed:
            # Calculate time until next admission window
            current_day = int(self.env.now // 24)
            current_hour = self.env.now % 24
            
            if self.night_admission_restriction and (current_hour < 8 or current_hour >= 17):
                # Wait until 8 AM the next day or same day
                next_allowed_time = (current_day * 24) + 8
                if current_hour >= 17:  # If it's evening, wait until next morning
                    next_allowed_time += 24
            elif self.weekend_admission_restriction and (int((self.env.now // 24) % 7) >= 5):
                # Wait until Monday 8 AM
                days_until_monday = 7 - int((self.env.now // 24) % 7)
                if days_until_monday == 7:
                    days_until_monday = 0
                next_allowed_time = ((current_day + days_until_monday) * 24) + 8
            else:
                # No restriction should be active at this point, but just in case
                next_allowed_time = self.env.now
            
            # Wait until admission is allowed
            wait_time = max(0, next_allowed_time - self.env.now)
            if wait_time > 0:
                yield self.env.timeout(wait_time)
        
        # Determine which resource to use
        bed_resource = self.crisis_bed_resource if is_crisis else self.regular_bed_resource
        
        # Time when the patient begins waiting for a bed
        bed_request_time = self.env.now
        
        # Request a bed
        with bed_resource.request() as request:
            # Wait until a bed becomes available
            yield request
            
            # Calculate waiting time for bed
            bed_waiting_time = self.env.now - bed_request_time
            
            # Apply triage delay (transfer time before actual admission)
            triage_delay = np.random.exponential(self.triage_delay_mean * 24)  # Convert days to hours
            yield self.env.timeout(triage_delay)
            
            # Calculate total waiting time including triage
            total_waiting_time = bed_waiting_time + triage_delay
            
            # Record waiting time statistics (if after warm-up period)
            if arrival_time >= self.warm_up_period * 24:
                if is_crisis:
                    self.stats['crisis_waiting_times'].append(total_waiting_time / 24)  # Convert to days
                else:
                    self.stats['regular_waiting_times'].append(total_waiting_time / 24)  # Convert to days
            
            # Patient is now admitted - determine length of stay
            los_mean = self.crisis_los_mean if is_crisis else self.regular_los_mean
            length_of_stay = np.random.gamma(shape=2, scale=los_mean/2) * 24  # Convert days to hours
            
            # Record patient admission details
            admission_time = self.env.now
            expected_discharge_time = admission_time + length_of_stay
            
            if arrival_time >= self.warm_up_period * 24:
                self.stats['patient_log'].append({
                    'patient_id': patient_id,
                    'type': 'Crisis' if is_crisis else 'Regular',
                    'arrival_time': arrival_time,
                    'admission_time': admission_time,
                    'waiting_time': total_waiting_time / 24,  # Convert to days
                    'expected_discharge_time': expected_discharge_time,
                    'expected_los': length_of_stay / 24  # Convert to days
                })
            
            # Occupy the bed for the entire length of stay
            yield self.env.timeout(length_of_stay)
            
            # Record discharge
            if self.env.now >= self.warm_up_period * 24:
                day = int(self.env.now // 24)
                self.stats['daily_discharges'][day] += 1

    def collect_occupancy_stats(self):
        """Periodically collect occupancy and queue statistics"""
        while True:
            # Wait for the next hour
            yield self.env.timeout(1)
            
            # Only record statistics after the warm-up period
            if self.env.now >= self.warm_up_period * 24:
                day = int(self.env.now // 24)
                
                # Record occupancy
                self.stats['regular_occupancy'][day] = self.regular_bed_resource.count
                self.stats['crisis_occupancy'][day] = self.crisis_bed_resource.count
                
                # Record queue lengths
                self.stats['regular_queue_length'][day] = len(self.regular_bed_resource.queue)
                self.stats['crisis_queue_length'][day] = len(self.crisis_bed_resource.queue)
    
    def run_simulation(self):
        """Run the simulation for the specified duration"""
        # Start processes
        self.env.process(self.patient_generator(is_crisis=False))  # Regular patient arrivals
        self.env.process(self.patient_generator(is_crisis=True))   # Crisis patient arrivals
        self.env.process(self.collect_occupancy_stats())           # Statistics collection
        
        # Run the simulation
        self.env.run(until=(self.warm_up_period + self.simulation_duration) * 24)
        
        # Process and return results
        return self.process_results()
    
    def process_results(self):
        """Process and summarize simulation results"""
        results = {}
        
        # Average waiting times
        results['avg_regular_waiting_time'] = np.mean(self.stats['regular_waiting_times']) if self.stats['regular_waiting_times'] else 0
        results['avg_crisis_waiting_time'] = np.mean(self.stats['crisis_waiting_times']) if self.stats['crisis_waiting_times'] else 0
        results['max_regular_waiting_time'] = np.max(self.stats['regular_waiting_times']) if self.stats['regular_waiting_times'] else 0
        results['max_crisis_waiting_time'] = np.max(self.stats['crisis_waiting_times']) if self.stats['crisis_waiting_times'] else 0
        
        # Create daily statistics dataframe
        days = range(self.warm_up_period, self.warm_up_period + self.simulation_duration)
        daily_stats = {
            'day': days,
            'regular_occupancy': [self.stats['regular_occupancy'].get(day, 0) for day in days],
            'crisis_occupancy': [self.stats['crisis_occupancy'].get(day, 0) for day in days],
            'regular_queue': [self.stats['regular_queue_length'].get(day, 0) for day in days],
            'crisis_queue': [self.stats['crisis_queue_length'].get(day, 0) for day in days],
            'regular_arrivals': [self.stats['regular_daily_arrivals'].get(day, 0) for day in days],
            'crisis_arrivals': [self.stats['crisis_daily_arrivals'].get(day, 0) for day in days],
            'discharges': [self.stats['daily_discharges'].get(day, 0) for day in days]
        }
        
        results['daily_stats'] = pd.DataFrame(daily_stats)
        
        # Calculate average occupancy rates
        results['avg_regular_occupancy'] = np.mean(daily_stats['regular_occupancy']) / self.regular_beds if self.regular_beds > 0 else 0
        results['avg_crisis_occupancy'] = np.mean(daily_stats['crisis_occupancy']) / self.crisis_beds if self.crisis_beds > 0 else 0
        results['avg_regular_queue'] = np.mean(daily_stats['regular_queue'])
        results['avg_crisis_queue'] = np.mean(daily_stats['crisis_queue'])
        
        # Patient log
        results['patient_log'] = pd.DataFrame(self.stats['patient_log'])
        
        return results
    
    def plot_results(self, results):
        """Create visualizations of simulation results"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Bed Occupancy Over Time
        daily_stats = results['daily_stats']
        axes[0, 0].plot(daily_stats['day'], daily_stats['regular_occupancy'], label='Regular Beds')
        axes[0, 0].plot(daily_stats['day'], daily_stats['crisis_occupancy'], label='Crisis Beds')
        axes[0, 0].axhline(y=self.regular_beds, color='r', linestyle='--', label='Regular Capacity')
        axes[0, 0].axhline(y=self.crisis_beds, color='g', linestyle='--', label='Crisis Capacity')
        axes[0, 0].set_title('Bed Occupancy Over Time')
        axes[0, 0].set_xlabel('Simulation Day')
        axes[0, 0].set_ylabel('Number of Beds Occupied')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Queue Lengths Over Time
        axes[0, 1].plot(daily_stats['day'], daily_stats['regular_queue'], label='Regular Queue')
        axes[0, 1].plot(daily_stats['day'], daily_stats['crisis_queue'], label='Crisis Queue')
        axes[0, 1].set_title('Queue Lengths Over Time')
        axes[0, 1].set_xlabel('Simulation Day')
        axes[0, 1].set_ylabel('Number of Patients Waiting')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Waiting Time Distribution
        if results['patient_log'].shape[0] > 0:
            axes[1, 0].hist(results['patient_log']['waiting_time'], bins=20, alpha=0.7)
            axes[1, 0].set_title('Distribution of Waiting Times')
            axes[1, 0].set_xlabel('Waiting Time (days)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)
        
        # Plot 4: Arrivals and Discharges
        axes[1, 1].plot(daily_stats['day'], daily_stats['regular_arrivals'], label='Regular Arrivals')
        axes[1, 1].plot(daily_stats['day'], daily_stats['crisis_arrivals'], label='Crisis Arrivals')
        axes[1, 1].plot(daily_stats['day'], daily_stats['discharges'], label='Discharges')
        axes[1, 1].set_title('Daily Patient Flow')
        axes[1, 1].set_xlabel('Simulation Day')
        axes[1, 1].set_ylabel('Number of Patients')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig

# Example usage:
def run_scenario(name, **params):
    """Run a scenario with specified parameters and print results"""
    print(f"\nRunning scenario: {name}")
    sim = IntermediateCareSimulation(**params)
    results = sim.run_simulation()
    
    print(f"Average Regular Waiting Time: {results['avg_regular_waiting_time']:.2f} days")
    print(f"Average Crisis Waiting Time: {results['avg_crisis_waiting_time']:.2f} days")
    print(f"Maximum Regular Waiting Time: {results['max_regular_waiting_time']:.2f} days")
    print(f"Maximum Crisis Waiting Time: {results['max_crisis_waiting_time']:.2f} days")
    print(f"Average Regular Bed Occupancy: {results['avg_regular_occupancy']*100:.1f}%")
    print(f"Average Crisis Bed Occupancy: {results['avg_crisis_occupancy']*100:.1f}%")
    print(f"Average Regular Queue Length: {results['avg_regular_queue']:.2f} patients")
    print(f"Average Crisis Queue Length: {results['avg_crisis_queue']:.2f} patients")
    
    # Plot and save results
    fig = sim.plot_results(results)
    fig.suptitle(f"Scenario: {name}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.savefig(f"{name.replace(' ', '_')}.png")
    
    return results, sim

if __name__ == "__main__":
    # Run baseline scenario
    baseline_results, baseline_sim = run_scenario(
        "Baseline", 
        regular_beds=50,
        crisis_beds=10,
        regular_arrival_rate=8,
        crisis_arrival_rate=2,
        regular_los_mean=14,
        crisis_los_mean=7,
        triage_delay_mean=1.5,
        time_dependent_arrivals=True,
        weekend_admission_restriction=True,
        night_admission_restriction=True,
        simulation_duration=365,
        warm_up_period=30
    )
    
    # Run improved triage scenario
    improved_triage_results, improved_triage_sim = run_scenario(
        "Improved Triage", 
        regular_beds=50,
        crisis_beds=10,
        regular_arrival_rate=8,
        crisis_arrival_rate=2,
        regular_los_mean=14,
        crisis_los_mean=7,
        triage_delay_mean=0.5,  # Reduced triage delay
        time_dependent_arrivals=True,
        weekend_admission_restriction=True,
        night_admission_restriction=True,
        simulation_duration=365,
        warm_up_period=30
    )
    
    # Run increased capacity scenario
    increased_capacity_results, increased_capacity_sim = run_scenario(
        "Increased Capacity", 
        regular_beds=60,  # 20% more regular beds
        crisis_beds=12,   # 20% more crisis beds
        regular_arrival_rate=8,
        crisis_arrival_rate=2,
        regular_los_mean=14,
        crisis_los_mean=7,
        triage_delay_mean=1.5,
        time_dependent_arrivals=True,
        weekend_admission_restriction=True,
        night_admission_restriction=True,
        simulation_duration=365,
        warm_up_period=30
    )
    
    # Run 24/7 admissions scenario
    extended_hours_results, extended_hours_sim = run_scenario(
        "24-7 Admissions", 
        regular_beds=50,
        crisis_beds=10,
        regular_arrival_rate=8,
        crisis_arrival_rate=2,
        regular_los_mean=14,
        crisis_los_mean=7,
        triage_delay_mean=1.5,
        time_dependent_arrivals=True,
        weekend_admission_restriction=False,  # No weekend restrictions
        night_admission_restriction=False,    # No night restrictions
        simulation_duration=365,
        warm_up_period=30
    )