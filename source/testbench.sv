`timescale 1ns / 1ps

module polynomial_approximation_tb;

// Inputs
reg [15:0] x;

// Outputs
wire [15:0] sin_x;
logic clock;
  always begin
    #2;
    clock = ~clock;
  end
// Instantiate the Unit Under Test (UUT)
polynomial_approximation uut (
    .x(x), 
    .sin_x(sin_x)
);

initial begin
    // Initialize Inputs
    x = 0;

    // Wait 100 ns for global reset to finish
    #100;

    // Add stimulus here
    // Test a range of values, for example from -π to π in fixed-point format
    // Note: Adjust the range and step as per your fixed-point format and requirements

    // Example: x = 0 (should result in sin_x close to 0)
    x = 16'd0; // Representing 0 in fixed-point
    #10; // Wait for the module to process

    // Example: x = π/2 (should result in sin_x close to 1)
    x = 16'd16384; // Representing π/2 in fixed-point (assuming π is 16'd32768)
    #10;

    // Continue with more test values as needed

    // Finish the simulation
    $finish;
end

endmodule

