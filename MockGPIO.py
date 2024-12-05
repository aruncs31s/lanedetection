class MockGPIO:
    BCM = "BCM"
    OUT = "OUT"
    LOW = "LOW"
    HIGH = "HIGH"

    @staticmethod
    def setmode(mode):
        print(f"Setting GPIO mode to {mode}")

    @staticmethod
    def setwarnings(flag):
        pass

    @staticmethod
    def setup(pin, mode):
        print(f"Setting up GPIO pin {pin} as {mode}")

    @staticmethod
    def output(pin, state):
        print(f"Setting GPIO pin {pin} to {state}")

    class PWM:
        def __init__(self, pin, frequency):
            print(f"Setting PWM on pin {pin} with frequency {frequency}")

        def start(self, duty_cycle):
            print(f"Starting PWM with duty cycle {duty_cycle}")

        def ChangeDutyCycle(self, duty_cycle):
            print(f"Changing PWM duty cycle to {duty_cycle}")

        def stop(self):
            print("Stopping PWM")
