module polynomial_approximation (
    input [15:0] x,  // 16-bit fixed point input
    output reg [15:0] sin_x  // 16-bit fixed point output
);

// Coefficients for the polynomial approximation, scaled for fixed-point arithmetic
// These values will change depending on your fixed-point format
parameter COEFF_1 = 16'd32767; // Coefficient for x^1, scaled
parameter COEFF_3 = -16'd5461; // Coefficient for x^3, scaled
parameter COEFF_5 = 16'd273;   // Coefficient for x^5, scaled

// Temporary variables for intermediate values
reg [31:0] temp1, temp2, temp3;

always @(x) begin
    // Polynomial evaluation using Horner's method
    // Assuming x is in the range -π to π and scaled to 16-bit fixed-point format
    temp1 = x * x;  // x^2
    temp2 = temp1 * x / 16'd6; // x^3 / 3!
    temp3 = temp2 * temp1 / 16'd120; // x^5 / 5!

    // Sum up the series
    sin_x = COEFF_1 * x + COEFF_3 * temp2 + COEFF_5 * temp3;
end

endmodule

