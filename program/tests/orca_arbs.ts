// import { Program } from '@project-serum/anchor';
// import { Tmp } from '../target/types/tmp';

// import * as web3 from '@solana/web3.js';
// import * as anchor from '@project-serum/anchor';
// import * as fs from 'fs';
// import { TOKEN_PROGRAM_ID } from "@solana/spl-token";
// import BN from 'bn.js';
// import { decode_poolparams, deriveAssociatedTokenAddress, get_balance } from './utils';
// import Decimal from "decimal.js";

// import * as orca_root from '../../orca_pools/typescript-sdk/src';
// import { Owner } from '../../orca_pools/typescript-sdk/src/public/utils/web3/key-utils';
// import { OrcaPoolParams } from '../../orca_pools/typescript-sdk/src/model/orca/pool/pool-types';

// const ORCA_TOKENSWAP_ID = new web3.PublicKey(
//   '8qEj6WU2gSGUHZdRTxSFUuYpU49BtfoQDfnZA6RWTEph'
// );

// async function pool_swap_ix(
//     program:Program<Tmp>,
//     owner:web3.Keypair, 
//     pool_params, 
//     zero_2_one:boolean
// ): Promise<web3.TransactionInstruction[]> {

//   var token0 = pool_params['tokens'][pool_params['tokenIds'][0]];
//   var token1 = pool_params['tokens'][pool_params['tokenIds'][1]];

//   var inputToken = zero_2_one ? token0 : token1;
//   var outputToken = zero_2_one ? token1 : token0;
//   console.log(`setting up swap for ... ${inputToken['tag']} -> ${outputToken['tag']}`)

//   let tokenSwap = pool_params['address'];
//   let userTransferAuthority = owner.publicKey;
//   let poolSource = inputToken['addr'];
//   let poolDestination = outputToken['addr'];
//   let poolMint = pool_params['poolTokenMint'];
//   let feeAccount = pool_params['feeAccount'];
//   let swapProgramId = ORCA_TOKENSWAP_ID;
//   let tokenProgramId = TOKEN_PROGRAM_ID;

//   const [authorityForPoolAddress, _] = await web3.PublicKey.findProgramAddress(
//     [tokenSwap.toBuffer()],
//     ORCA_TOKENSWAP_ID
//   );
//   let authority = authorityForPoolAddress;

//   let accountA = await deriveAssociatedTokenAddress(
//     owner.publicKey, 
//     inputToken['mint']
//   )
//   let accountB = await deriveAssociatedTokenAddress(
//     owner.publicKey, 
//     outputToken['mint']
//   )
//   let userSource = accountA;
//   let userDestination = accountB;
  
//   const [information_pda, sb] = await anchor.web3.PublicKey.findProgramAddress(
//     [Buffer.from("information")],
//     program.programId
//   );

//   // A -> B
//   const ix = program.instruction.orcaSwap({
//     accounts: {
//       tokenSwap: tokenSwap,
//       authority: authority,
//       userTransferAuthority: userTransferAuthority,
//       userSrc: userSource, 
//       poolSrc: poolSource, 
//       userDst: userDestination, 
//       poolDst: poolDestination,
//       poolMint: poolMint, 
//       feeAccount: feeAccount,
//       tokenProgram: tokenProgramId,
//       tokenSwapProgram: swapProgramId,
//       information: information_pda,
//     },
//     signers: [owner],
//   });

//   return [ix]
// }

// describe('tmp', async () => {
  
//   // Configure the client to use the local cluster.
//   const provider = anchor.Provider.env();
//   const connection = provider.connection;
//   anchor.setProvider(provider);
  
//   const program = anchor.workspace.Tmp as Program<Tmp>;
  
//   // load the owner of the tokens 
//   let rawdata = fs.readFileSync(
//     `/Users/brennan/Documents/workspace/solana/orca_local/env/pools/owner.key`, 
//     'utf8'
//   );  
//   let owner_secret = new Uint8Array(JSON.parse(rawdata));
//   let owner = web3.Keypair.fromSecretKey(owner_secret);

//   // setup PDAs 
//   const [information_pda, sb] = await anchor.web3.PublicKey.findProgramAddress(
//       [Buffer.from("information")],
//       program.programId
//   );

//   let info = await connection.getAccountInfo(information_pda);
  
//   if (info == null) {
//     var init_tx = await program.rpc.initProgram({
//       accounts: {
//         information: information_pda, 
//         payer: provider.wallet.publicKey,
//         systemProgram: web3.SystemProgram.programId
//       },
//     })
//     console.log('init pda...', init_tx);
//   } else { 
//     console.log('pda already initialized...');
//   }

//   const orca = orca_root.getOrca(connection, ORCA_TOKENSWAP_ID)

//   // load the pool info for the swap 
//   var pool_path = '/Users/brennan/Documents/workspace/solana/orca_local/env/pools/params_AB.json'
//   const AB_pool_params = decode_poolparams(pool_path)
//   var pool_path = '/Users/brennan/Documents/workspace/solana/orca_local/env/pools/params_BC.json'
//   const BC_pool_params = decode_poolparams(pool_path)
//   var pool_path = '/Users/brennan/Documents/workspace/solana/orca_local/env/pools/params_CA.json'
//   const CA_pool_params = decode_poolparams(pool_path)

//   const pools = [
//     AB_pool_params, 
//     BC_pool_params, 
//     CA_pool_params,
//   ]
//   let name2token = {}
  
//   for (let pool of pools) {
//     var token0 = pool['tokens'][pool['tokenIds'][0]];
//     name2token[token0.name] = token0

//     var token1 = pool['tokens'][pool['tokenIds'][1]];
//     name2token[token1.name] = token1
//   }
//   let tokenA = name2token['A']
//   let tokenB = name2token['B']
//   let tokenC = name2token['C']

//   let accountA = await deriveAssociatedTokenAddress(
//     owner.publicKey, 
//     tokenA['mint']
//   )
//   let accountB = await deriveAssociatedTokenAddress(
//     owner.publicKey, 
//     tokenB['mint']
//   )
//   let accountC = await deriveAssociatedTokenAddress(
//     owner.publicKey, 
//     tokenC['mint']
//   )
//   // //8HQpT4EnjPWMxsqJozAUC5GukwyPTngm14gcufozaPTG
//   // //9BKLeyHiS942NdfBcFWE3XAkenRybgqoYXE12GghYEh3
//   // console.log(accountA.toString(), accountB.toString())

//   let token2pool: { [id: string] : orca_root.OrcaPool; } = {}
//   let token2poolparams: { [id: string] : OrcaPoolParams; } = {}
//   let tokens: orca_root.OrcaPoolToken[] = []
//   let token_names: string[] = []

//   for (let pool_params of pools) {
//     const pool = orca.getPool(pool_params);
//     let tokenA = pool.getTokenA();
//     let tokenB = pool.getTokenB();
    
//     // track the tokens 
//     if (!token_names.includes(tokenA.name)) {
//         tokens.push(tokenA)
//         token_names.push(tokenA.name)
//     }
//     if (!token_names.includes(tokenB.name)) {
//         tokens.push(tokenB)
//         token_names.push(tokenB.name)
//     }
//     // track the tokens -> pools 
//     let id = tokenA.name + tokenB.name;
//     token2pool[id] = pool;
//     token2poolparams[id] = pool_params;
//   }  
//   const n_tokens = tokens.length; 
//   console.log(tokens.map(t => t.name))

//   // const pool = orca.getPool(AB_pool_params);
//   // let quote: orca_root.Quote = await pool.getQuote(pool.getTokenA(), new Decimal(100));
//   // v = quote.getExpectedOutputAmount().toNumber();
//   // console.log(`Swap 1 ${pool.getTokenA().tag} for ${v} ${pool.getTokenB().tag}`);

//   // create a graph of the pools 
//   let graph = []
//   for (let i = 0; i < n_tokens; i++) {
//     let row = []
//     for (let j = 0; j < n_tokens; j++) {
//       row.push(0)
//     }
//     graph.push(row)
//   }

//   // fill in the graph with exchange rates 
//   var i = 0 
//   for (i = 0; i < n_tokens; i++) {
//       for (let j = i; j < n_tokens; j++) {
//           if (i == j) { 
//               graph[i][j] = 1
//           } else {
//               let tokenA: orca_root.OrcaToken = tokens[i]
//               let tokenB: orca_root.OrcaToken = tokens[j]

//               var id = tokenA.name + tokenB.name;
//               if (!(id in token2pool)) {
//                   id = tokenB.name + tokenA.name;
//               }
              
//               var v; 
//               if (!(id in token2pool)) {
//                   // invalid pool / cannot swap / no edge 
//                   graph[i][j] = 0 // will be Infinity 
//                   graph[j][i] = 0 // will be Infinity when -log(0)
//                   console.log(`No Pool For ${tokenA.tag} and ${tokenB.tag}`);
//               } else {
//                   // assumes balance = 1 
//                   // need to find optimal balance for swaps for there still to be 
//                   // a negative weight cycle -- but for every possible pool? 
//                   let pool: orca_root.OrcaPool = token2pool[id];
//                   let quote: orca_root.Quote = await pool.getQuote(tokenA, new Decimal(1));
//                   // v = quote.getMinOutputAmount().toNumber();
//                   v = quote.getExpectedOutputAmount().toNumber();
//                   console.log(`Swap 1 ${tokenA.tag} for ${v} ${tokenB.tag}`);
//                   graph[i][j] = v
//                   graph[j][i] = 1/v
//               }
//           }
//       }
//   }

//   // neglog 
//   i = 0;
//   var j = 0;
//   for (i = 0; i < n_tokens; i++) {
//       for (j = 0; j < n_tokens; j++) {
//           graph[i][j] = -Math.log(graph[i][j])
//       }
//   }

//   // bellman-ford algo 
//   let n = n_tokens; 
//   let dist = []
//   let prev = []
//   for (i=0; i < n; i++) {
//       dist.push(Infinity)
//       prev.push(-1)
//   }
//   dist[0] = 0 // distance from starting node 
  
//   var count;
//   for (count=0; count < n - 1; count++) {
//     for (i=0; i < n; i++) {
//       for (j=0; j < n; j++) {
//         var score = 0;
//         score = dist[i] + graph[i][j];
//         if (score < dist[j]) {
//           dist[j] = score 
//           prev[j] = i
//         }
//       }
//     }
//   }

//   let arb_paths = []
//   for (i=0; i < n; i++) {
//     for (j=0; j < n; j++) {
//       var score = 0;
//       score = dist[i] + graph[i][j];
//       if (score < dist[j]) {
//           let cycle = [j, i];

//           let k = i
//           while (!cycle.includes(prev[k])) {
//               cycle.push(prev[k])
//               k = prev[k]
//           }
//           cycle.push(prev[k])
//           cycle = cycle.reverse()

//           // only care about negative cycles which start and finish 
//           // at the same node 
//           if (cycle[0] == cycle[cycle.length-1]) {
//               console.log('--- FOUND ARB! ---')
//               cycle.forEach(idx => {
//                   console.log('\t ->', tokens[idx].name)
//               })
//               arb_paths.push(cycle)
//           }
//       }
//     }
//   }
//   console.log(arb_paths); // [[1, 2, 0, 1], [2, 1, 2]]

//   // EARLY EXIT IF NO ARBS FOUND 
//   if (arb_paths.length == 0) return []

//   // find the BEST arb path 
//   let max_balance = -Infinity
//   let max_balance_idx = -1
//   for (let j=0; j < arb_paths.length; j++) {
//       let arb_path = arb_paths[j];
//       balance = 1
//       for (i=0; i < arb_path.length-1; i++) {
//           var token0_idx = arb_path[i];
//           var token1_idx = arb_path[i+1];
//           let ex_rate = Math.exp(-graph[token0_idx][token1_idx])
//           balance *= ex_rate; 
//       }
//       console.log(`Arb Path ${j} = 1 -> ${balance}`)
//       if (balance > max_balance) {
//           max_balance = balance
//           max_balance_idx = j;
//       }
//   }

//   // visualize it 
//   console.log('--- ARB VISUAL ---')
//   let arb_path = arb_paths[max_balance_idx]
//   let init_balance = 1; 

//   var balance = init_balance; 
//   var token_name = tokens[arb_path[0]].name;
//   for (i=0; i < arb_path.length-1; i++) {
//       var token0_idx = arb_path[i];
//       var token1_idx = arb_path[i+1];
//       let ex_rate = Math.exp(-graph[token0_idx][token1_idx])

//       var token0_name = tokens[token0_idx].name;
//       var token1_name = tokens[token1_idx].name;
      
//       console.log(`\t BALANCE: ${balance} ${token_name}`)
//       console.log(`\t EXCHANGE RATE: 1 ${token0_name} = ${ex_rate} ${token1_name}`)
      
//       balance *= ex_rate; 
//       token_name = token1_name;
//       console.log(`\t-> NEW BALANCE: ${balance} ${token_name}`)
      
//       console.log(`\t----`)
//   }
//   console.log('')

//   // setting up the trade 
//   let init_token: orca_root.OrcaToken = tokens[arb_path[0]]
//   var init_token_account_addr = await orca_root.deriveAssociatedTokenAddress(owner.publicKey, init_token.mint);
//   var _start_balance = (await get_balance(connection, init_token_account_addr)) * 10 ** init_token.scale
  
//   // gather swap ixs 
//   var ixs = []
//   let amount_in = 1 * 10 ** 2; // 1 token with a 2 decimal mint 
//   // record amount_in of FIRST swap 
//   var start_ix = program.instruction.startSwap(new anchor.BN(amount_in), {
//     accounts: {
//       information: information_pda, 
//     }
//   });
//   ixs.push(start_ix)

//   for (i=0; i < arb_path.length-1; i++) {
//       var idx0 = arb_path[i];
//       var idx1 = arb_path[i+1];
      
//       let token0 = tokens[idx0];
//       let token1 = tokens[idx1];

//       var id:string = token0.name + token1.name;
//       var zero_to_one = true;
//       if (!(id in token2pool)) {
//           id = token1.name + token0.name;
//           zero_to_one = false; 
//       }
//       let pool_params = token2poolparams[id]

//       var ixsss = await pool_swap_ix(program, owner, pool_params, zero_to_one);
//       ixs.push(...ixsss)      
//   }

//   var ix = program.instruction.profitOrRevert(new anchor.BN(_start_balance), {
//     accounts: {
//       src: init_token_account_addr,
//       information: information_pda,
//     }
//   })
//   ixs.push(ix)
  
//   const recentBlockHash = (await connection.getRecentBlockhash("singleGossip")).blockhash;
//   const txFields: web3.TransactionCtorFields = {
//     recentBlockhash: recentBlockHash,
//     feePayer: owner.publicKey,
//   };
//   const transaction = new web3.Transaction(txFields)
//     .add(...ixs);

//   const signers = [owner];

//   console.log('Balance: A, B, C:', 
//     await get_balance(connection, accountA), 
//     await get_balance(connection, accountB),
//     await get_balance(connection, accountC),
//   )

//   let tx = await web3.sendAndConfirmTransaction(provider.connection, transaction, signers);
//   console.log('swap tx ...', tx);

//   console.log('Balance: A, B, C:', 
//     await get_balance(provider.connection, accountA), 
//     await get_balance(provider.connection, accountB),
//     await get_balance(provider.connection, accountC),
//   )
// });
