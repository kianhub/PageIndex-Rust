//! ID-token validation against OpenAI's JWKS (RS256 public client).

use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use serde::Deserialize;
use serde_json::Value;

use crate::error::{ErrorKind, PageIndexError, Result};

use super::ISSUER;

#[derive(Debug, Clone, Deserialize)]
pub struct IdClaims {
    pub sub: String,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub nonce: Option<String>,
}

#[derive(Deserialize)]
struct Jwks {
    keys: Vec<Jwk>,
}

#[derive(Deserialize)]
struct Jwk {
    kty: String,
    #[serde(default)]
    kid: Option<String>,
    #[serde(default)]
    n: Option<String>,
    #[serde(default)]
    e: Option<String>,
}

pub async fn verify_id_token(
    http: &reqwest::Client,
    jwks_url: &str,
    id_token: &str,
    issued_client_id: &str,
    expected_nonce: Option<&str>,
) -> Result<IdClaims> {
    let header = decode_header(id_token)
        .map_err(|e| PageIndexError::new(ErrorKind::Api, format!("invalid ID token header: {e}")))?;
    if header.alg != Algorithm::RS256 {
        return Err(PageIndexError::new(ErrorKind::Api, format!("ID token alg {:?} is not RS256", header.alg)));
    }
    let jwks = fetch_jwks(http, jwks_url).await?;
    let jwk = select_jwk(&jwks, header.kid.as_deref())?;
    let key = rsa_key(jwk)?;
    let mut validation = Validation::new(Algorithm::RS256);
    validation.set_issuer(&[ISSUER]);
    validation.set_audience(&[issued_client_id]);
    validation.set_required_spec_claims(&["sub", "exp", "iat"]);
    validation.leeway = 5;
    let token = decode::<IdClaims>(id_token, &key, &validation)
        .map_err(|e| PageIndexError::new(ErrorKind::Api, format!("ID token validation failed: {e}")))?;
    if let Some(expected) = expected_nonce
        && token.claims.nonce.as_deref() != Some(expected)
    {
        return Err(PageIndexError::new(ErrorKind::Api, "The ID token nonce did not match."));
    }
    if token.claims.sub.is_empty() {
        return Err(PageIndexError::new(ErrorKind::Api, "The ID token did not contain a subject."));
    }
    Ok(token.claims)
}

async fn fetch_jwks(http: &reqwest::Client, jwks_url: &str) -> Result<Jwks> {
    let response = http
        .get(jwks_url)
        .send()
        .await
        .map_err(|e| PageIndexError::new(ErrorKind::Api, format!("JWKS request failed: {e}")))?;
    let status = response.status();
    let payload: Value = response.json().await.unwrap_or(Value::Null);
    if !status.is_success() {
        return Err(
            PageIndexError::new(ErrorKind::Api, format!("JWKS returned {status}")).with_status(Some(status.as_u16()))
        );
    }
    serde_json::from_value(payload).map_err(|e| PageIndexError::new(ErrorKind::Api, format!("invalid JWKS: {e}")))
}

fn select_jwk<'a>(jwks: &'a Jwks, kid: Option<&str>) -> Result<&'a Jwk> {
    if let Some(kid) = kid {
        if let Some(jwk) = jwks.keys.iter().find(|k| k.kid.as_deref() == Some(kid)) {
            return Ok(jwk);
        }
        return Err(PageIndexError::new(ErrorKind::Api, "ID token kid was not present in JWKS"));
    }
    match jwks.keys.as_slice() {
        [one] => Ok(one),
        _ => Err(PageIndexError::new(ErrorKind::Api, "ID token header omitted kid and JWKS is ambiguous")),
    }
}

fn rsa_key(jwk: &Jwk) -> Result<DecodingKey> {
    if jwk.kty != "RSA" {
        return Err(PageIndexError::new(ErrorKind::Api, format!("unsupported JWK kty {}", jwk.kty)));
    }
    let n = jwk.n.as_deref().ok_or_else(|| PageIndexError::new(ErrorKind::Api, "RSA JWK missing n"))?;
    let e = jwk.e.as_deref().ok_or_else(|| PageIndexError::new(ErrorKind::Api, "RSA JWK missing e"))?;
    DecodingKey::from_rsa_components(n, e)
        .map_err(|e| PageIndexError::new(ErrorKind::Api, format!("invalid RSA JWK: {e}")))
}
