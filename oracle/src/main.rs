//! Synthetic stationary CHIMERA reference, no external audio or identities.
//! stdin: {anatomy:[scale,pharynx,radius,width],artic:[17],source:[flow samples],fs:Hz,freqs:[Hz]}
//! stdout: geometry, linear elements, transfer magnitude, and actual AirwayNetwork::step audio.
use chimera::anatomy::human::{HumanArtic, HumanBody};
use chimera::airway::frequency::transfer_function;
use chimera::voice::human_airway::HumanAirway;
use serde_json::{Value, json};
use std::io::{self, Read};

fn numbers(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(|x| x.as_f64().expect("finite number")).collect()
}

fn glottis_params(values: &[f64]) -> chimera::sources::glottis::GlottisParams {
    assert_eq!(values.len(),17);
    chimera::sources::glottis::GlottisParams {
        f0:values[0], amplitude:values[1], abduction:values[2], chink:values[3],
        open_quotient:values[4], skew:values[5], closure_sharpness:values[6], effort:values[7],
        jitter:values[8], shimmer:values[9], vibrato_depth:values[10], vibrato_rate:values[11],
        drift:values[12], fry:values[13], diplophonia:values[14], closure:values[15], chaos:values[16]
    }
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let v: Value = serde_json::from_str(&input).unwrap();
    if v["operation"] == "glottis" {
        let fs = v["fs"].as_f64().unwrap();
        let mut glottis = chimera::sources::glottis::ParametricGlottis::new(fs);
        glottis.params = glottis_params(&numbers(&v["glottis"]));
        let dp = numbers(&v["pressure"]);
        let areas: Vec<f64> = dp.iter().map(|p| glottis.next(*p)).collect();
        println!("{}",json!({"area":areas}));
        return;
    }
    let body = numbers(&v["anatomy"]);
    let mut a = numbers(&v["artic"]);
    let source = numbers(&v["source"]);
    let freqs = numbers(&v["freqs"]);
    let fs = v["fs"].as_f64().unwrap();
    assert_eq!(body.len(), 4);
    assert_eq!(a.len(), 17);
    assert!(fs > 0.0);
    let body = HumanBody { name: "custom", scale: body[0], pharynx: body[1], tongue_radius: body[2], lip_width: body[3] };
    let mut airway = HumanAirway::new(fs, body);
    for (value, spec) in a.iter_mut().zip(chimera::control::articulators::ARTIC_SPECS) {
        *value = value.clamp(spec.min, spec.max);
    }
    let artic = HumanArtic::from_array(a.as_slice().try_into().unwrap());
    airway.apply(&artic, 0.0);
    airway.net.snap_to_targets();
    let el = airway.net.linear_elements();
    let transfer = transfer_function(&el, airway.pharynx.first_node(), &freqs, &[airway.glottis_port]);
    // This oracle tests the documented *linear*, stationary ideal-flow contract.
    // Silence the turbulence and wall-motion sources. Tiny external flow makes
    // nonlinear Bernoulli losses negligible, while retaining the actual Rust
    // integrator, walls, radiation, and geometry. No HumanVoice substitutions.
    let noise_enabled = v["noise"].as_bool().unwrap_or(false);
    if !noise_enabled { airway.net.set_noise_sources(&[]); }
    let gesture = v["operation"] == "gesture";
    airway.net.enable_volume_source = gesture;
    let frames: Vec<HumanArtic> = if gesture {
        v["artic_frames"].as_array().unwrap().iter().map(|frame| {
            let values = numbers(frame);
            HumanArtic::from_array(values.as_slice().try_into().unwrap())
        }).collect()
    } else { vec![artic] };
    let frame_size = v["frame_size"].as_u64().unwrap_or(1) as usize;
    let noise_specs: Vec<Value> = airway.net.noise_sources().iter().map(|s|json!({
        "src":s.src,"fwd":s.fwd,"bwd":s.bwd,"gain":s.gain
    })).collect();
    let mut glottis = chimera::sources::glottis::ParametricGlottis::new(fs);
    if v["operation"] == "coupled" || gesture {
        glottis.params = glottis_params(&numbers(&v["glottis"]));
        airway.net.set_fixed_pressure(airway.lungs, v["lung_pressure"].as_f64().unwrap()*980.665);
    }
    let mut glottal_area = Vec::new();
    let mut glottal_flow = Vec::new();
    let mut audio = Vec::with_capacity(source.len());
    for (i,sample) in source.iter().enumerate() {
        if gesture && i % frame_size == 0 {
            let index = (i/frame_size+1).min(frames.len()-1);
            airway.apply(&frames[index],0.0);
            if !noise_enabled { airway.net.set_noise_sources(&[]); }
            airway.net.begin_block(frame_size);
        }
        if v["operation"] == "coupled" || gesture {
            let area = glottis.next(airway.net.p[airway.trachea.last_node()]-airway.net.p[airway.pharynx.first_node()]);
            airway.net.set_port_area(airway.glottis_port, area);
            glottal_area.push(area);
        } else {
            airway.net.ext_flow[airway.pharynx.first_node()] = *sample;
        }
        audio.push(airway.net.step());
        glottal_flow.push(airway.net.u[airway.glottis_port]);
    }
    println!("{}", json!({
        "fs":fs, "anatomy":body, "artic":a, "source":source, "freqs":freqs,
        "ph":airway.ph_areas, "orl":airway.or_areas, "ph_len":airway.ph_len,
        "or_len":airway.or_len, "ph_min":airway.ph_min, "or_min":airway.or_min,
        "teeth_s":airway.frame.as_ref().unwrap().teeth_s, "velum":airway.frame.as_ref().unwrap().velum_port,
        "a":el.a,"b":el.b,"l":el.l,"r":el.r,"open":el.open,"c":el.c,
        "wall_area":el.wall_area,"rad_area":el.rad_area,"mag":transfer.mag,"audio":audio, "glottal_area":glottal_area, "glottal_flow":glottal_flow, "noise_specs":noise_specs
    }));
}
