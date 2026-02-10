import { Program } from '@project-serum/anchor';
import { Tmp } from '../target/types/tmp';

import * as anchor from "@project-serum/anchor";
import { assert } from "chai";
import * as web3 from "@solana/web3.js";
import * as token from "@solana/spl-token";

const fs = require("fs");

// import { decode_poolparams, decode_pool_base, deriveAssociatedTokenAddress, get_balance } from './utils';
const ORCA_TOKENSWAP_PID = new web3.PublicKey(
    '9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP'
);
const MERCURIAL_PID = new web3.PublicKey(
    'MERLuDFBMmsHnsBPZw2sDQZHvXFMwp8EdjudcU2HKky'
);

describe("tmp", () => {
    let provider = anchor.Provider.env();
    let connection = provider.connection;

    web3.LAMPORTS_PER_SOL

    anchor.setProvider(provider);
    const program = anchor.workspace.Tmp as Program<Tmp>;

    let rawdata = fs.readFileSync(`../mainnet-fork/localnet_owner.key`, 'utf8');  
    let owner_secret = new Uint8Array(JSON.parse(rawdata));
    let wallet = web3.Keypair.fromSecretKey(owner_secret);

    it("gets some WSOL", async () => {
        // WSOL ATA
        let src_token_account = (await web3.PublicKey.findProgramAddress(
            [wallet.publicKey.toBuffer(), token.TOKEN_PROGRAM_ID.toBuffer(), token.NATIVE_MINT.toBuffer()],
            token.ASSOCIATED_TOKEN_PROGRAM_ID
        ))[0];
        
        let amount_in = 1 * web3.LAMPORTS_PER_SOL;
        // go SOL -> WSOL 
        let ixs = [
            web3.SystemProgram.transfer({
                fromPubkey: wallet.publicKey,
                toPubkey: src_token_account,
                lamports: amount_in,
            }),    
            // @ts-ignore
            token.createSyncNativeInstruction(src_token_account)
        ]
        const recentBlockHash = (await connection.getRecentBlockhash("singleGossip")).blockhash;
        const txFields: web3.TransactionCtorFields = {
            recentBlockhash: recentBlockHash,
            feePayer: wallet.publicKey,
        };
        let tx = new web3.Transaction(txFields);
        tx = tx.add(...ixs);

        let amount0 = (await connection.getTokenAccountBalance(src_token_account)).value.uiAmount
        if (amount0 < 0.1) {
            await web3.sendAndConfirmTransaction(connection, tx, [wallet])
            let amount1 = (await connection.getTokenAccountBalance(src_token_account)).value.uiAmount
    
            console.log(amount0, amount1)
            assert(amount1 > amount0);
        } else {
            console.log("already has WSOL:", amount0);
        }
    });

    it("sets up the info pda", async () => {
        const [state_pda, sb] = await anchor.web3.PublicKey.findProgramAddress(
            [Buffer.from("swap_state")],
            program.programId
        );    
        console.log("pda:", state_pda.toString())
        let info = await connection.getAccountInfo(state_pda);
        if (info == null) {
            console.log("initializing pda...")
            await program.rpc.initProgram({
                accounts: {
                    swapState: state_pda, 
                    payer: provider.wallet.publicKey,
                    systemProgram: web3.SystemProgram.programId,
                }
            });
        } else { 
            console.log("pda already initialized...")
        }
    })
    
    return; 

    // NOTE: the rest (pool-specific swap instructions) are tested using rust code 

    // ---- graveyard of dead code ----
    //              r.i.p
    //       long live rust code

    // it("does an orca swap", async () => {
    //     const [information_pda, sb] = await anchor.web3.PublicKey.findProgramAddress(
    //         [Buffer.from("information")],
    //         program.programId
    //     );    
    //     let pool_path = "../orca_pools/mainnet_pools/pools/params_SOL_USDC.json";
    //     let pool_params = decode_poolparams(pool_path)

    //     let amount_in = 0.1 * web3.LAMPORTS_PER_SOL;

    //     let input_token = pool_params.tokens["So11111111111111111111111111111111111111112"]
    //     let output_token = pool_params.tokens["EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"]

    //     let src_token_account = (await web3.PublicKey.findProgramAddress(
    //         [wallet.publicKey.toBuffer(), token.TOKEN_PROGRAM_ID.toBuffer(), input_token.mint.toBuffer()],
    //         token.ASSOCIATED_TOKEN_PROGRAM_ID
    //     ))[0];
    //     let dst_token_account = (await web3.PublicKey.findProgramAddress(
    //         [wallet.publicKey.toBuffer(), token.TOKEN_PROGRAM_ID.toBuffer(), output_token.mint.toBuffer()],
    //         token.ASSOCIATED_TOKEN_PROGRAM_ID
    //     ))[0];

    //     await program.rpc.startSwap(new anchor.BN(amount_in), {
    //         accounts: {
    //             information: information_pda
    //         }
    //     });

    //     let tokenSwap = pool_params['address'];
    //     let userTransferAuthority = wallet.publicKey;
    //     let poolSource = input_token['addr'];
    //     let poolDestination = output_token['addr'];
    //     let poolMint = pool_params['poolTokenMint'];
    //     let feeAccount = pool_params['feeAccount'];
    //     let swapProgramId = ORCA_TOKENSWAP_PID;
    //     let tokenProgramId = token.TOKEN_PROGRAM_ID;
    //     const [authority, _] = await web3.PublicKey.findProgramAddress(
    //       [tokenSwap.toBuffer()],
    //       ORCA_TOKENSWAP_PID
    //     );

    //     let b0_src = (await connection.getTokenAccountBalance(src_token_account)).value.uiAmountString;
    //     let b0_dst = (await connection.getTokenAccountBalance(dst_token_account)).value.uiAmountString;

    //     await program.rpc.orcaSwap({
    //         accounts: {
    //             tokenSwap: tokenSwap,
    //             authority: authority,
    //             userTransferAuthority: userTransferAuthority,
    //             userSrc: src_token_account, 
    //             poolSrc: poolSource, 
    //             userDst: dst_token_account, 
    //             poolDst: poolDestination,
    //             poolMint: poolMint, 
    //             feeAccount: feeAccount,
    //             tokenProgram: tokenProgramId,
    //             tokenSwapProgram: swapProgramId,
    //             information: information_pda,
    //         },
    //         signers: [wallet]
    //     });

    //     let b1_src = (await connection.getTokenAccountBalance(src_token_account)).value.uiAmountString;
    //     let b1_dst = (await connection.getTokenAccountBalance(dst_token_account)).value.uiAmountString;
        
    //     console.log(b0_src, b1_src)
    //     console.log(b0_dst, b1_dst)
        
    //     assert(b0_src > b1_src); // out 
    //     assert(b1_dst > b0_dst); // in 
    // });
    

    // it("does an saber swap", async () => {
    //     const [information_pda, sb] = await anchor.web3.PublicKey.findProgramAddress(
    //         [Buffer.from("information")],
    //         program.programId
    //     );    
    //     let pool_path = "../saber_sdk/pools/4EFeyTtMZZnAv3ZPs2jvps1T1J1JykbpyizrWjBQcA1S_saber_pool.json";
    //     let pool_params = decode_pool_base(pool_path)

    //     let amount_in = 0.1 * web3.LAMPORTS_PER_SOL;

    //     let input_token = pool_params['token_ids'][0];
    //     let output_token = pool_params['token_ids'][1];

    //     let in_mint = pool_params['tokens'][input_token]["mint"]
    //     let out_mint = pool_params['tokens'][output_token]["mint"]

    //     let src_token_account = (await web3.PublicKey.findProgramAddress(
    //         [wallet.publicKey.toBuffer(), token.TOKEN_PROGRAM_ID.toBuffer(), in_mint.toBuffer()],
    //         token.ASSOCIATED_TOKEN_PROGRAM_ID
    //     ))[0];
    //     let dst_token_account = (await web3.PublicKey.findProgramAddress(
    //         [wallet.publicKey.toBuffer(), token.TOKEN_PROGRAM_ID.toBuffer(), out_mint.toBuffer()],
    //         token.ASSOCIATED_TOKEN_PROGRAM_ID
    //     ))[0];

    //     await program.rpc.startSwap(new anchor.BN(amount_in), {
    //         accounts: {
    //             information: information_pda
    //         }
    //     });

    //     const SWAP_PROGRAM_ID = new web3.PublicKey("SSwpkEEcbUqx4vtoEByFjSkhKdCT862DNVb52nZg1UZ");

    //     let b0_src = (await connection.getTokenAccountBalance(src_token_account)).value.uiAmountString;
    //     let b0_dst = (await connection.getTokenAccountBalance(dst_token_account)).value.uiAmountString;

    //     await program.rpc.saberSwap({
    //         accounts: {
    //             poolAccount: pool_params["pool_account"],
    //             authority: pool_params['authority'], 
    //             userTransferAuthority: wallet.publicKey, 
    //             userSrc: src_token_account, 
    //             userDst: dst_token_account, 
    //             poolSrc: pool_params['tokens'][input_token]["addr"],
    //             poolDst: pool_params['tokens'][output_token]["addr"],
    //             feeDst: pool_params["fee_accounts"][output_token],
    //             saberSwapProgram: SWAP_PROGRAM_ID, 
    //             information: information_pda, 
    //             tokenProgram: token.TOKEN_PROGRAM_ID,
    //         },
    //         signers: [wallet]
    //     });

    //     let b1_src = (await connection.getTokenAccountBalance(src_token_account)).value.uiAmountString;
    //     let b1_dst = (await connection.getTokenAccountBalance(dst_token_account)).value.uiAmountString;
        
    //     console.log("src balance:", b0_src, "->", b1_src)
    //     console.log("dst balance:", b0_dst, "->", b1_dst)
        
    //     assert(b0_src > b1_src); // out 
    //     assert(b1_dst > b0_dst); // in 

    // });

    // it("does a round-trip mercurial swap", async () => {
    //     const [information_pda, sb] = await anchor.web3.PublicKey.findProgramAddress(
    //         [Buffer.from("information")],
    //         program.programId
    //     );    
        
    //     // rando pool
    //     const POOL_ACCOUNT = new web3.PublicKey("MAR1zHjHaQcniE2gXsDptkyKUnNfMEsLBVcfP7vLyv7");
    //     let stableSwapNPool = await StableSwapNPool.load(connection, POOL_ACCOUNT, wallet.publicKey)

    //     // WSOL ATA & mSOL ATA
    //     let src_token_account = new web3.PublicKey("3armvRWvS8Xhiz1Xe3WGwPnVACYPVXpDzfr7wqfnvYNt");
    //     let dst_token_account = new web3.PublicKey("23fSTDw6TcR1jdAjKHfFr3s2rRAVefs6QY4oNjHGWYnw");

    //     let amount_in = 0.1 * web3.LAMPORTS_PER_SOL;

    //     var amount0_src = (await connection.getTokenAccountBalance(src_token_account)).value.uiAmount
    //     var amount0_dst = (await connection.getTokenAccountBalance(dst_token_account)).value.uiAmount
        
    //     await program.rpc.startSwap(new anchor.BN(amount_in), {
    //         accounts: {
    //             information: information_pda
    //         }
    //     });

    //     await program.rpc.mercurialSwap({
    //         accounts: {
    //             poolAccount: stableSwapNPool.poolAccount,
    //             authority: stableSwapNPool.authority,
    //             userTransferAuthority: wallet.publicKey,
    //             userSrc: src_token_account, 
    //             userDst: dst_token_account, 
    //             poolSrc: stableSwapNPool.tokenAccounts[0],
    //             poolDst: stableSwapNPool.tokenAccounts[1],
    //             tokenProgram: token.TOKEN_PROGRAM_ID,
    //             mercurialSwapProgram: MERCURIAL_PID, 
    //             information: information_pda,
    //         },
    //         signers: [wallet]
    //     });

    //     var amount1_src = (await connection.getTokenAccountBalance(src_token_account)).value.uiAmount
    //     var amount1_dst = (await connection.getTokenAccountBalance(dst_token_account)).value.uiAmount
        
    //     console.log("A -> B")
    //     console.log("src:", amount0_src, amount1_src)
    //     console.log("dst:", amount0_dst, amount1_dst)

    //     // swap?!
    //     assert(amount0_src > amount1_src)
    //     assert(amount0_dst < amount1_dst)

    //     // swap back! -- amount out of prev trade will already be recorded in PDA 
    //     var amount0_src = (await connection.getTokenAccountBalance(src_token_account)).value.uiAmount
    //     var amount0_dst = (await connection.getTokenAccountBalance(dst_token_account)).value.uiAmount
    //     await program.rpc.mercurialSwap({
    //         accounts: {
    //             poolAccount: stableSwapNPool.poolAccount,
    //             authority: stableSwapNPool.authority,
    //             userTransferAuthority: wallet.publicKey,
    //             userSrc: dst_token_account, // only change these? yes.
    //             userDst: src_token_account, // only change these? yes.
    //             poolSrc: stableSwapNPool.tokenAccounts[0],
    //             poolDst: stableSwapNPool.tokenAccounts[1],
    //             tokenProgram: token.TOKEN_PROGRAM_ID,
    //             mercurialSwapProgram: MERCURIAL_PID, 
    //             information: information_pda,
    //         },
    //         signers: [wallet]
    //     });
    //     var amount1_src = (await connection.getTokenAccountBalance(src_token_account)).value.uiAmount
    //     var amount1_dst = (await connection.getTokenAccountBalance(dst_token_account)).value.uiAmount
        
    //     console.log("B -> A")
    //     console.log("src:", amount0_src, amount1_src)
    //     console.log("dst:", amount0_dst, amount1_dst)
        
    //     assert(amount0_src < amount1_src)
    //     assert(amount0_dst > amount1_dst)
    // })
    
});
