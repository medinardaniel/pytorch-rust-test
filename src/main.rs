
fn main() -> anyhow::Result<()> {
    println!("Starting model inference...");
    // declare a random 1x1024 input tensor
    let input = tch::Tensor::randn(&[1, 1024], tch::kind::FLOAT_CPU);
    // load the model
    let model = tch::CModule::load("model.pt")?;
    // get output
    let output = model.forward_ts(&[input])?;

    println!("Model output: {:?}", output);
    Ok(())
}
