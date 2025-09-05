// import * as web3 from '@solana/web3.js';
// import * as fs from 'fs';
// import { TOKEN_PROGRAM_ID } from "@solana/spl-token";
// import * as spl_token from "@solana/spl-token";

// // import * as orca_root from '../../orca_pools/typescript-sdk/src';
// // import { OrcaPoolParams } from '../../orca_pools/typescript-sdk/src/model/orca/pool/pool-types';

// export async function get_balance(connection: web3.Connection, addr: web3.PublicKey): Promise<number> {
//   let b = await connection.getTokenAccountBalance(addr)  
//   return b.value.uiAmount as number
// }

// export function decode_pool_base(path: string) {
//   let rawdata = fs.readFileSync(path, 'utf8');
//   let poolParams = JSON.parse(rawdata);
//   // save pool information 
//   function decode_key2base58(d: { [id: string] : string; } , newd: { [id: string] : string | web3.PublicKey | {}; } ): {} {
//     for (var key in d) {
//       let v = d[key];
//       if (v.constructor.name == 'Object') {
//         newd[key] = {};
//         newd[key] = decode_key2base58(v as {}, {})
//       } else {
//         if (typeof(v) == 'string' && v.length > 10) { // pubkey string ? 
//           newd[key] = new web3.PublicKey(v);
//         } else {
//           newd[key] = v;
//         }
//       }
//     }
//     return newd
//   }
//   let pool_params = decode_key2base58(poolParams, {})
//   return pool_params
// }

// export function decode_poolparams(path: string): OrcaPoolParams {
//   let rawdata = fs.readFileSync(path, 'utf8');
//   let poolParams = JSON.parse(rawdata);
//   // save pool information 
//   function decode_key2base58(d: { [id: string] : string; } , newd: { [id: string] : string | web3.PublicKey | {}; } ): {} {
//     for (var key in d) {
//       let v = d[key];
//       if (v.constructor.name == 'Object') {
//         newd[key] = {};
//         newd[key] = decode_key2base58(v as {}, {})
//       } else {
//         if (typeof(v) == 'string' && v.length > 10) { // pubkey string ? 
//           newd[key] = new web3.PublicKey(v);
//         } else {
//           newd[key] = v;
//         }
//       }
//     }
//     return newd
//   }
//   let pool_params = decode_key2base58(poolParams, {}) as OrcaPoolParams;
  
//   // manually create feestructure percentages 
//   pool_params.feeStructure.traderFee = orca_root.Percentage.fromFraction(
//     parseInt(pool_params.feeStructure.traderFee.numerator as unknown as string), 
//     parseInt(pool_params.feeStructure.traderFee.denominator as unknown as string), 
//   )
//   pool_params.feeStructure.ownerFee = orca_root.Percentage.fromFraction(
//     parseInt(pool_params.feeStructure.ownerFee.numerator as unknown as string), 
//     parseInt(pool_params.feeStructure.ownerFee.denominator as unknown as string), 
//   )
//   return pool_params
// }

// export async function deriveAssociatedTokenAddress(
//   walletAddress: web3.PublicKey,
//   tokenMint: web3.PublicKey
// ): Promise<web3.PublicKey> {
//   return (
//     await web3.PublicKey.findProgramAddress(
//       [walletAddress.toBuffer(), TOKEN_PROGRAM_ID.toBuffer(), tokenMint.toBuffer()],
//       spl_token.ASSOCIATED_TOKEN_PROGRAM_ID
//     )
//   )[0];
// }
