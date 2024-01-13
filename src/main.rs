use std::collections::HashMap;

use bumpalo::collections::{String as BumpString, Vec as BumpVec};
use bumpalo::Bump;

// ============================================================================
// Abstract Syntax Tree

#[derive(Debug)]
enum Ast<'a> {
    // Literal(&'a Literal<'a>),
    BodyStmt(BumpVec<'a, Stmt<'a>>),
}

#[derive(Debug)]
enum Literal<'a> {
    Num(isize),
    Str(BumpString<'a>),
}

#[derive(Debug)]
enum Stmt<'a> {
    Var(&'a VarStmt<'a>),
    Expr(&'a Expr<'a>),
}

/// Variable declaration statement (and optionally defined).
#[derive(Debug)]
struct VarStmt<'a> {
    name: String,
    rhs: Option<&'a Expr<'a>>,
}

#[derive(Debug)]
enum Expr<'a> {
    /// Variable access by name.
    Access(BumpString<'a>),
    Literal(&'a Literal<'a>),
    Binary(&'a BinaryExpr<'a>),
}

#[derive(Debug)]
struct BinaryExpr<'a> {
    op: BinaryOperator,
    lhs: &'a Expr<'a>,
    rhs: &'a Expr<'a>,
}

#[derive(Debug)]
enum BinaryOperator {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

#[derive(Debug)]
struct SyntaxTree<'a> {
    root: Ast<'a>,
    bump: &'a Bump,
}

trait IntoLiteral {
    fn into_literal(self, bump: &Bump) -> &mut Literal;
}

impl IntoLiteral for &str {
    #[inline(always)]
    fn into_literal(self, bump: &Bump) -> &mut Literal {
        bump.alloc(Literal::Str(BumpString::from_str_in(self, bump)))
    }
}

impl IntoLiteral for isize {
    #[inline(always)]
    fn into_literal(self, bump: &Bump) -> &mut Literal {
        bump.alloc(Literal::Num(self))
    }
}

trait IntoExpr<'a> {
    fn into_expr(self, bump: &'a Bump) -> &'a mut Expr<'a>;
}

impl<'a> IntoExpr<'a> for &'a Literal<'a> {
    #[inline(always)]
    fn into_expr(self, bump: &'a Bump) -> &'a mut Expr<'a> {
        bump.alloc(Expr::Literal(self))
    }
}

impl<'a> IntoExpr<'a> for &'a mut Literal<'a> {
    #[inline(always)]
    fn into_expr(self, bump: &'a Bump) -> &'a mut Expr<'a> {
        bump.alloc(Expr::Literal(self))
    }
}

impl<'a> IntoExpr<'a> for &'a BinaryExpr<'a> {
    #[inline(always)]
    fn into_expr(self, bump: &'a Bump) -> &'a mut Expr<'a> {
        bump.alloc(Expr::Binary(self))
    }
}

impl<'a> IntoExpr<'a> for &'a mut BinaryExpr<'a> {
    #[inline(always)]
    fn into_expr(self, bump: &'a Bump) -> &'a mut Expr<'a> {
        bump.alloc(Expr::Binary(self))
    }
}

impl<'a> IntoExpr<'a> for &'a mut Expr<'a> {
    #[inline(always)]
    fn into_expr(self, _: &'a Bump) -> &'a mut Expr<'a> {
        self
    }
}

// ============================================================================
// Pretty Print

struct PPrint<'a> {
    ast: &'a Ast<'a>,
}

impl<'a> PPrint<'a> {
    fn fmt_literal(f: &mut std::fmt::Formatter, lit: &Literal) -> std::fmt::Result {
        match *lit {
            Literal::Num(n) => write!(f, "{n}"),
            Literal::Str(ref s) => write!(f, "{s:?}"),
        }
    }

    fn fmt_binary_op(f: &mut std::fmt::Formatter, op: &BinaryOperator) -> std::fmt::Result {
        match op {
            BinaryOperator::Add => write!(f, "+"),
            BinaryOperator::Sub => write!(f, "-"),
            BinaryOperator::Mul => write!(f, "*"),
            BinaryOperator::Div => write!(f, "/"),
            BinaryOperator::Mod => write!(f, "%"),
        }
    }

    fn fmt_binary_expr(f: &mut std::fmt::Formatter, binary_expr: &BinaryExpr) -> std::fmt::Result {
        Self::fmt_expr(f, binary_expr.lhs)?;
        write!(f, " ")?;
        Self::fmt_binary_op(f, &binary_expr.op)?;
        write!(f, " ")?;
        Self::fmt_expr(f, binary_expr.rhs)?;

        Ok(())
    }

    fn fmt_expr(f: &mut std::fmt::Formatter, expr: &Expr) -> std::fmt::Result {
        match *expr {
            Expr::Access(ref name) => write!(f, "{name}"),
            Expr::Literal(lit) => Self::fmt_literal(f, lit),
            Expr::Binary(binary_expr) => Self::fmt_binary_expr(f, binary_expr),
        }
    }
}

impl<'a> std::fmt::Display for PPrint<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.ast {
            Ast::BodyStmt(stmts) => {
                for stmt in stmts {
                    match *stmt {
                        Stmt::Var(var_stmt) => {
                            let name = var_stmt.name.as_str();

                            match var_stmt.rhs {
                                Some(rhs) => {
                                    write!(f, "var {name} = ")?;
                                    Self::fmt_expr(f, rhs)?;
                                    writeln!(f)?;
                                }
                                None => {
                                    writeln!(f, "var {name}")?;
                                }
                            }
                        }
                        Stmt::Expr(expr) => {
                            Self::fmt_expr(f, expr)?;
                            writeln!(f)?;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Builder Functions
//
// When a `struct Builder<'a>` type is used, the functions taking `&self`
// bind the returned AST nodes to the builder's instance. It becomes impossible
// to return the built nodes because Rust thinks they're borrowing `&Builder`.

#[inline(always)]
fn literal(bump: &Bump, value: impl IntoLiteral) -> &mut Literal {
    value.into_literal(bump)
}

#[inline(always)]
fn access_var<'a>(bump: &'a Bump, name: &str) -> &'a mut Expr<'a> {
    let string = BumpString::from_str_in(name, bump);
    bump.alloc(Expr::Access(string))
}

#[inline(always)]
fn var_def<'a>(bump: &'a Bump, name: String, rhs: impl IntoExpr<'a>) -> &'a mut VarStmt {
    bump.alloc(VarStmt {
        name,
        rhs: Some(rhs.into_expr(bump)),
    })
}

#[inline(always)]
fn var_decl<'a>(bump: &'a Bump, name: String) -> &'a mut VarStmt {
    bump.alloc(VarStmt { name, rhs: None })
}

#[inline(always)]
fn binary_expr<'a>(
    bump: &'a Bump,
    op: BinaryOperator,
    lhs: impl IntoExpr<'a>,
    rhs: impl IntoExpr<'a>,
) -> &'a mut BinaryExpr {
    bump.alloc(BinaryExpr {
        op,
        lhs: lhs.into_expr(bump),
        rhs: rhs.into_expr(bump),
    })
}

// ============================================================================
// "Parser"

fn parse(bump: &Bump) -> SyntaxTree {
    let mut stmts: BumpVec<Stmt> = BumpVec::new_in(bump);

    // 3 + 7
    {
        let literal1 = literal(bump, 3);
        let literal2 = literal(bump, 7);
        let binary_expr = binary_expr(bump, BinaryOperator::Add, literal1, literal2);
        let expr = binary_expr.into_expr(bump);

        stmts.push(Stmt::Expr(expr));
    }

    // var x = 11 + (13 + 17)
    {
        let literal1 = literal(bump, 11);
        let literal2 = literal(bump, 13);
        let literal3 = literal(bump, 17);
        let binary_expr1 = binary_expr(bump, BinaryOperator::Add, literal2, literal3);
        let binary_expr2 = binary_expr(bump, BinaryOperator::Add, literal1, binary_expr1);
        let var_stmt = var_def(bump, "x".to_string(), binary_expr2);

        stmts.push(Stmt::Var(var_stmt));
    }

    // var y
    {
        let var_stmt = var_decl(bump, "y".to_string());

        stmts.push(Stmt::Var(var_stmt));
    }

    // var z = x + y
    {
        let access1 = access_var(bump, "x");
        let access2 = access_var(bump, "y");
        let binary_expr = binary_expr(bump, BinaryOperator::Add, access1, access2);
        let var_stmt = var_def(bump, "z".to_string(), binary_expr);

        stmts.push(Stmt::Var(var_stmt));
    }

    let root = Ast::BodyStmt(stmts);

    SyntaxTree { root, bump }
}

// ============================================================================
// Opcodes

/// # Binary Operations
///
/// Binary operations have a third argument for the destination register
/// where the resulting value will be stored.
///
/// This might seem superfluous when the LHS or RHS are temporary values,
/// because they can be resused. However they might be local variables in
/// which case the arithmetic operation must not overwrite the variable.
#[derive(Debug)]
#[allow(non_camel_case_types)]
enum Op {
    Add(u8, u8, u8),
    Sub,
    Mul,
    Div,
    Mod,
    LoadConstant_Number(u8, isize),
    LoadConstant_String(u8, String),
    /// Store a null/nil value in the given register.
    StoreZero(u8),
}

// ============================================================================
// Compiler

/// Maximum number of registers that a function can allocate.
///
/// Limited by the argument space in in the instruction set.
const MAX_REGISTERS: usize = u8::MAX as usize;

struct Compiler {
    code: Vec<Op>,
    locals: HashMap<String, Reg>,
    /// Keeps track of which registers are allocated
    /// for the current function.
    reg: Registers,
}

type CompileResult<T> = Result<T, String>;

impl Compiler {
    fn new() -> Self {
        Self {
            code: vec![],
            locals: HashMap::new(),
            reg: Registers::new(),
        }
    }

    /// Reset compiler to avoid left over state leaks,
    fn reset(&mut self) {
        self.reg.clear();
    }

    fn compile(&mut self, tree: &SyntaxTree) -> CompileResult<Vec<Op>> {
        self.compile_ast(&tree.root)?;
        println!("function needs {} registers", self.reg.reg_space);

        println!("======\nlocals\n======\n");
        for (name, reg) in self.locals.iter() {
            println!("  {name} : {}", reg.id);
        }
        println!("\n");

        let code = self.code.drain(..).collect();
        self.reset();

        Ok(code)
    }

    /// Compile an abstract-syntax-tree.
    fn compile_ast(&mut self, ast: &Ast) -> CompileResult<()> {
        match *ast {
            Ast::BodyStmt(ref stmts) => self.compile_body_stmts(stmts.as_slice()),
        }
    }

    /// Compile a block body.
    fn compile_body_stmts(&mut self, stmts: &[Stmt]) -> CompileResult<()> {
        for stmt in stmts {
            self.compile_stmt(stmt)?;
        }

        Ok(())
    }

    /// Compile a simple statement.
    fn compile_stmt(&mut self, stmt: &Stmt) -> CompileResult<()> {
        match *stmt {
            Stmt::Var(var_stmt) => {
                self.compile_var_stmt(var_stmt)?;
            }
            Stmt::Expr(expr) => {
                let reg = self.compile_expr(expr)?;

                // Expression statement discards its final result to
                // free up the register.
                self.reg.release_temp(&reg);
            }
        }

        Ok(())
    }

    /// Compile a variable declaration/definition statement.
    fn compile_var_stmt(&mut self, var_stmt: &VarStmt) -> CompileResult<()> {
        let reg = match var_stmt.rhs {
            // Statement has assignment.
            Some(expr) => {
                // The temporary register can be used as a local variable.
                let mut reg = self.compile_expr(expr)?;
                assert!(self.reg.is_acquired(reg.id), "invariant: when a compiler function returns a register, that register must not be released");
                reg.kind = RegKind::Local;
                self.reg.patch(&reg)?;
                reg
            }
            // Variable is only declared but not defined.
            None => {
                let reg = self.reg.acquire(RegKind::Local)?;
                self.emit(Op::StoreZero(reg.id));
                reg
            }
        };

        self.locals.insert(var_stmt.name.clone(), reg);

        Ok(())
    }

    /// Compile an expression.
    ///
    /// Returns the register where the resulting value will be stored.
    fn compile_expr(&mut self, expr: &Expr) -> CompileResult<Reg> {
        match *expr {
            Expr::Access(ref name) => self.compile_access_expr(name.as_str()),
            Expr::Literal(lit) => self.compile_literal_expr(lit),
            Expr::Binary(binary_expr) => self.compile_binary_expr(binary_expr),
        }
    }

    /// Compile a binary expression.
    fn compile_binary_expr(&mut self, expr: &BinaryExpr) -> CompileResult<Reg> {
        let reg1 = self.compile_expr(&expr.lhs)?;
        let reg2 = self.compile_expr(&expr.rhs)?;

        // The VM will load these registers onto the Rust stack,
        // so they can be reused to store the resulting value.
        let reg3 = if reg1.kind.is_temp() {
            self.reg.release_temp(&reg2);
            reg1.clone()
        } else if reg2.kind.is_temp() {
            self.reg.release_temp(&reg1);
            reg2.clone()
        } else {
            self.reg.acquire(RegKind::Temp)?
        };

        match expr.op {
            BinaryOperator::Add => self.emit(Op::Add(reg1.id, reg2.id, reg3.id)),
            BinaryOperator::Sub => self.emit(Op::Sub),
            BinaryOperator::Mul => self.emit(Op::Mul),
            BinaryOperator::Div => self.emit(Op::Div),
            BinaryOperator::Mod => self.emit(Op::Mod),
        }

        Ok(reg3)
    }

    /// Compile literal value expression.
    fn compile_literal_expr(&mut self, lit: &Literal) -> CompileResult<Reg> {
        let reg = self.reg.acquire(RegKind::Temp)?;

        match *lit {
            Literal::Num(n) => self.emit(Op::LoadConstant_Number(reg.id, n)),
            Literal::Str(ref s) => self.emit(Op::LoadConstant_String(reg.id, s.to_string())),
        }

        Ok(reg)
    }

    /// Compile an expression that accesses a variable by name.
    fn compile_access_expr(&mut self, name: &str) -> CompileResult<Reg> {
        self.locals
            .get(name)
            .cloned()
            .ok_or_else(|| format!("variable '{name}' does not exist"))
    }

    fn emit(&mut self, op: Op) {
        self.code.push(op)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegKind {
    Local,
    Temp,
}

impl RegKind {
    fn is_temp(&self) -> bool {
        matches!(self, Self::Temp)
    }
}

/// Represents a register allocation.
#[derive(Debug, Clone)]
struct Reg {
    id: u8,
    kind: RegKind,
}

#[derive(Debug)]
struct Registers {
    reg: Box<[Option<RegKind>; MAX_REGISTERS]>,
    /// Tracks the highest register used for the function.
    ///
    /// Will be used by the VM to allocate register space
    /// on the operand stack for the function's call frame.
    reg_space: usize,
}

impl Registers {
    fn new() -> Self {
        Self {
            reg: Box::new([None; MAX_REGISTERS]),
            reg_space: 0,
        }
    }

    fn clear(&mut self) {
        self.reg.fill(None);
    }

    fn is_acquired(&self, id: u8) -> bool {
        self.reg
            .get(id as usize)
            .map(|reg| reg.is_some())
            .unwrap_or(false)
    }

    fn acquire(&mut self, kind: RegKind) -> CompileResult<Reg> {
        // Find first register that is vacant, or allocate a new one.
        match self.reg.iter().position(|reg| reg.is_none()) {
            Some(idx) => {
                if idx > MAX_REGISTERS {
                    return Err("register index out of bounds".to_string());
                }
                self.reg[idx] = Some(kind);
                self.reg_space = self.reg_space.max(idx + 1);
                Ok(Reg {
                    id: idx as u8,
                    kind,
                })
            }
            None => Err("maximum registers have been allocated for the function".to_string()),
        }
    }

    fn release(&mut self, reg_idx: usize) -> Option<RegKind> {
        let existing = self.reg[reg_idx];
        self.reg[reg_idx] = None;
        existing
    }

    /// Patches a register with the new given kind.
    ///
    /// The use case is to ellevate a temporary register to a local variable.
    ///
    /// # Errors
    ///
    /// Returns an error if the register has not been acquired yet.
    fn patch(&mut self, reg: &Reg) -> CompileResult<()> {
        let maybe_kind = &mut self.reg[reg.id as usize];

        match maybe_kind {
            Some(kind) => {
                *kind = reg.kind;
                Ok(())
            }
            None => Err(format!(
                "cannot patch register {}, it has not been allocated",
                reg.id
            )),
        }
    }

    /// Releases the given register if it's a temporary.
    fn release_temp(&mut self, reg: &Reg) {
        if reg.kind.is_temp() {
            self.release(reg.id as usize);
        }
    }
}

fn main() {
    println!("Starting...");
    let bump = Bump::new();

    let tree = parse(&bump);
    println!("=======\nparse()\n=======\n\n{:#?}", tree);

    let pprint = PPrint { ast: &tree.root };
    println!("====\ncode\n====\n\n{pprint}");

    let mut compiler = Compiler::new();
    let bytecode = compiler.compile(&tree).unwrap();
    println!("========\nbytecode\n========\n\n");
    for (idx, op) in bytecode.iter().enumerate() {
        println!("{idx:>6} : {op:?}");
    }
    println!("\n");

    println!("Done.")
}
